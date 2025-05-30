/** \class SBSScalerEvtHandler
    \ingroup Base
\brief Event handler for Hall C scalers
This class does the following
For a particular set of event types (here, event type 0)
decode the scalers and put some variables into global variables.
The global variables can then appear in the Podd output tree T.
In addition, a tree "TS" is created by this class; it contains
just the scaler data by itself.  Note, the "fName" is concatenated
with "TS" to ensure the tree is unqiue; further, "fName" is
concatenated with the name of the global variables, for uniqueness.
The list of global variables and how they are tied to the
scaler module and channels is defined here; eventually this
will be modified to use a scaler.map file
NOTE: if you don't have the scaler map file (e.g. Leftscalevt.map)
there will be no variable output to the Trees.
To use in the analyzer, your setup script needs something like this
~~~
     hndlr = new SBSScalerEvtHandler("sbs","SBS Scaler bank"));
     hndlr->AddEvType(1); // Repeat for each event type with scaler banks
     hndlr->SetUseFirstEvent(kTRUE);
     gHaEvtHandlers->Add (hndlr);
~~~
To enable debugging, add the line
~~~
     hndlr->SetDebugFile("Scalerdebug.txt");
~~~
\author  E. Brash based on THaScalerEvtHandler by R. Michaels
         S. Wood modified for SBS
*/

#include "THaEvtTypeHandler.h"
#include "SBSScalerEvtHandler.h"
#include "GenScaler.h"
#include "Scaler3800.h"
#include "Scaler3801.h"
#include "Scaler1151.h"
#include "Scaler560.h"
//#include "Scaler9001.h"
//#include "Scaler9250.h"
#include "THaAnalyzer.h"
#include "THaCodaData.h"
#include "THaEvData.h"
//#include "THcParmList.h"
//#include "THcGlobals.h"
#include "THaGlobals.h"
#include "TNamed.h"
#include "TMath.h"
#include "TString.h"
#include "TROOT.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <iterator>
#include "THaVarList.h"
#include "VarDef.h"
#include "Helper.h"
#include "TH1D.h"

using namespace std;
using namespace Decoder;

static const UInt_t ICOUNT    = 1;
static const UInt_t IRATE     = 2;
static const UInt_t ICURRENT = 3;
static const UInt_t ICHARGE   = 4;
static const UInt_t ITIME   = 5;
static const UInt_t ICUT = 6;
static const UInt_t MAXCHAN   = 32;
static const UInt_t defaultDT = 4;

SBSScalerEvtHandler::SBSScalerEvtHandler(const char *name, const char* description)
  : THaEvtTypeHandler(name,description),
    fBCM_Gain(0), fBCM_Offset(0), fBCM_SatOffset(0), fBCM_SatQuadratic(0), fBCM_delta_charge(0),
    evcount(0), evcountR(0.0), ifound(0), fNormIdx(-1),
    fNormSlot(-1),
    dvars(0),dvars_prev_read(0), dvarsFirst(0), fScalerTree(0), fUseFirstEvent(kTRUE),
    fOnlySyncEvents(kFALSE), fOnlyBanks(kFALSE), fDelayedType(-1),
    fClockChan(-1), fLastClock(0), fClockOverflows(0),fPhysicsEventNumber(-1)
{
  fRocSet.clear();
  fModuleSet.clear();
  scal_prev_read.clear();
  scal_present_read.clear();
  scal_overflows.clear();
  fHistosInitialized = false;
}

SBSScalerEvtHandler::~SBSScalerEvtHandler()
{
  // The tree object is owned by ROOT since it gets associated wth the output
  // file, so DO NOT delete it here. 
  if (!TROOT::Initialized()) {
    delete fScalerTree;
  }
  DeleteContainer(scalers);
  DeleteContainer(scalerloc);
  delete [] dvars_prev_read;
  delete [] dvars;
  delete [] dvarsFirst;
  delete [] fBCM_Gain;
  delete [] fBCM_Offset;
  delete [] fBCM_SatOffset;
  delete [] fBCM_SatQuadratic;
  delete [] fBCM_delta_charge;

  for( vector<UInt_t*>::iterator it = fDelayedEvents.begin();
       it != fDelayedEvents.end(); ++it )
    delete [] *it;
  fDelayedEvents.clear();
}

Int_t SBSScalerEvtHandler::Begin( THaRunBase* rb )
{
  THaEvtTypeHandler::Begin( rb );
  if( !fHistosInitialized ){
    fHistosInitialized = true;
    fIunserVsTime = new TH1D("fIunserVsTime", ";time (s);", 5000, 0, 5000);
    fIu1VsTime    = new TH1D("fIu1VsTime", ";time (s);", 5000, 0, 5000);
    fIunewVsTime  = new TH1D("fIunewVsTime", ";time (s);", 5000, 0, 5000);
    fIdnewVsTime  = new TH1D("fIdnewVsTime", ";time (s);", 5000, 0, 5000);
    fId1VsTime    = new TH1D("fId1VsTime", ";time (s);", 5000, 0, 5000);
    fId3VsTime    = new TH1D("fId3VsTime", ";time (s);", 5000, 0, 5000);
    fId10VsTime   = new TH1D("fId10VsTime", ";time (s);", 5000, 0, 5000);
  }
  return 0;
}

Int_t SBSScalerEvtHandler::End( THaRunBase* )
{
  // Process any delayed events in order received

  cout << "SBSScalerEvtHandler::End Analyzing " << fDelayedEvents.size() << " delayed scaler events" << endl;
  for(std::vector<UInt_t*>::iterator it = fDelayedEvents.begin();
      it != fDelayedEvents.end(); ++it) {
    UInt_t* rdata = *it;
    AnalyzeBuffer(rdata,kFALSE);
  }
  if (fDebugFile) *fDebugFile << "scaler tree ptr  "<<fScalerTree<<endl;
  // evNumber += 1;
  evNumberR = evNumber;
  if (fScalerTree) fScalerTree->Fill();

  for( vector<UInt_t*>::iterator it = fDelayedEvents.begin();
       it != fDelayedEvents.end(); ++it )
    delete [] *it;
  fDelayedEvents.clear();

  if (fScalerTree) fScalerTree->Write();
  
  double Ntrigs, NtrigsA, Time, BeamCurrent, BeamCharge, LiveTime;
  double clk_cnt = 0, clk_rate = 0, edtm_cnt = 0, unew_cnt = 0, d3_cnt = 0, d10_cnt = 0;
  
  if(fIunserVsTime!=NULL) fIunserVsTime->Write( 0, kOverwrite );
  if(fIu1VsTime!=NULL)    fIu1VsTime->Write( 0, kOverwrite );
  if(fIunewVsTime!=NULL)  fIunewVsTime->Write( 0, kOverwrite );
  if(fIdnewVsTime!=NULL)  fIdnewVsTime->Write( 0, kOverwrite );
  if(fId1VsTime!=NULL)    fId1VsTime->Write( 0, kOverwrite );
  if(fId3VsTime!=NULL)    fId3VsTime->Write( 0, kOverwrite );
  if(fId10VsTime!=NULL)   fId10VsTime->Write( 0, kOverwrite );
  
  THaAnalyzer* analyzer = THaAnalyzer::GetInstance();
  if(analyzer!=nullptr){// check that the analyzer actually exists... otherwise, skip
    const char* summaryfilename = analyzer->GetSummaryFileName();
    cout << "SBSScalerEvtHandler Summary in " << summaryfilename << endl;
    if( strcmp(summaryfilename,"")!=0  ) {
      ofstream ostr(summaryfilename, std::ofstream::app);
      if( ostr ) {
	// Write to file via cout
	//streambuf* cout_buf = cout.rdbuf();
	//cout.rdbuf(ostr.rdbuf());
	TDatime now;
	ostr << "SBS scalers Summary " //<< fRun->GetNumber()
	     << " completed " << now.AsString()
	     << endl << " count " << evcount << endl
	     << endl;
	
	for (UInt_t i = 0; i < scalerloc.size(); i++) {
	  TString name = scalerloc[i]->name; 
	  //tinfo = name + "/D";
	  //fScalerTree->Branch(name.Data(), &dvars[i], tinfo.Data(), 4000);
	  bool found = false;
	  if(name.Contains("L1A")){
	    found = true;
	    if(name.Contains("scaler") && !name.Contains("Rate"))NtrigsA = dvars[i];
	  }
	  if(name.Contains("BBCALTRG")){
	    found = true;
	    if(name.Contains("scaler") && !name.Contains("Rate"))Ntrigs = dvars[i];
	  }
	  if(name.Contains("EDTM")){
	    found = true;
	    if(name.Contains("scaler") && !name.Contains("Rate"))edtm_cnt = dvars[i];
	  }
	  if(name.Contains("104kHz_CLK")){
	    found = true;
	    cout << name.Data() << endl;
	    if(name.Contains("rate")){
	      cout << "104kHz clock rate? " << dvars[i] << endl;
	      clk_rate = dvars[i];
	    }else if(name.Contains("cnt")){
	      clk_cnt = dvars[i];
	    }
	  }
	  if(name.Contains("bcm")){
	    found = true;
	    if(name.Contains("unew.cnt") && !name.Contains("rate"))unew_cnt = dvars[i];
	    if(name.Contains("d3.cnt") && !name.Contains("rate"))d3_cnt = dvars[i];
	    if(name.Contains("d10.cnt") && !name.Contains("rate"))d10_cnt = dvars[i];
	  }
	  	  
	  if(found)ostr << " Scaler " << name.Data() <<  " value: " << dvars[i] << endl;
	}	
	//std::vector<Decoder::GenScaler*> scalers;
	//std::vector<ScalerVar*> scalerloc;
	ostr << endl;
	
	double unew_charge = unew_cnt/2.8725e3;//uC/counts  //TODO put those numbers in DB
	double d3_charge = d3_cnt/1.7e3;//uC/counts  //TODO put those numbers in DB
	double d10_charge = d10_cnt/7.52e3;//uC/counts  //TODO put those numbers in DB
	// those numbers have been obtained using run 11991 by plotting the 
	// corresponding scaler rates for this run and dividing by 
	// the beam current for this run i.e. 4uA
		
	Time = clk_cnt/clk_rate;
	BeamCharge = (unew_charge+d3_charge+d10_charge)/3;
	BeamCurrent = BeamCharge/Time;
	LiveTime = (edtm_cnt/Time)/21.;//TODO, put EDTM frequency in a DB
	//setting 21.0 Hz instead of 20.0 is sort of an educated guess 

	ostr << " scaler summary : N_trigs = " << Ntrigs << endl;
	ostr << " scaler summary : N_trigs_accepted = " << NtrigsA << endl;
	ostr << " scaler summary : Time = " << Time  << " s " << endl;
	ostr << " scaler summary : beam charge (unew) = " << unew_charge << " uC " << endl;
	ostr << " scaler summary : beam charge (d3) = " << d3_charge << " uC " << endl;
	ostr << " scaler summary : beam charge (d10) = " << d10_charge << " uC " << endl;
	ostr << " scaler summary : Average beam charge = " << BeamCharge << " uC " << endl;
	ostr << " scaler summary : Average beam current = " << BeamCurrent << " uA " << endl;
	ostr << " scaler summary : Live time (NtrigA/Ntrig) = " << NtrigsA/Ntrigs*100 << " % " << endl;
	ostr << " scaler summary : Live time (EDTM) = " << LiveTime*100 << " % " << endl;
	
	//cout.rdbuf(cout_buf);
	ostr.close();
	
      }
      
    }
  }
  
  return 0;
}


Int_t SBSScalerEvtHandler::ReadDatabase(const TDatime& date )
{
  char prefix[2];
  prefix[0]='g';
  prefix[1]='\0';
  fNumBCMs = 0;
// #ifdef HALLCPARM

  DBRequest list [] = { 
     {"NumBCMs",&fNumBCMs,kInt,0,1}, 
     {0} 
  };
     
  TString sname = "db_sbsBCM.dat"; 
  std::cout << "Trying to load database file " << sname << std::endl;

  // FILE* file = OpenFile( date );
  FILE *file = Podd::OpenDBFile(sname.Data(), date);

  if( !file ){
     std::cout << "*** ERROR! Cannot load DB file! ***" << std::endl;
     return kInitError;
  }

  Int_t err = kOK; 

  if(!err){
     err = LoadDB( file, date,list,fPrefix);
     if(err!=0) std::cout << "*** ERROR! Cannot load DB! ***" << std::endl;
  }

  // DBRequest list[]={
  //   {"NumBCMs",&fNumBCMs, kInt, 0, 1},
  //   {0}
  // };
  // gHcParms->LoadParmValues((DBRequest*)&list, prefix);
  cout << " Number of BCMs = " << fNumBCMs << endl;
  
  if(fNumBCMs > 0) {
    fBCM_Gain = new Double_t[fNumBCMs];
    fBCM_Offset = new Double_t[fNumBCMs];
    fBCM_SatOffset = new Double_t[fNumBCMs];
    fBCM_SatQuadratic = new Double_t[fNumBCMs];
    fBCM_delta_charge= new Double_t[fNumBCMs];
    string bcm_namelist;
    DBRequest list2[]={
      {"BCM_Names"                  , &bcm_namelist,                 kString},
      {"BCM_Gain"                   , fBCM_Gain,                     kDouble,  (UInt_t) fNumBCMs},
      {"BCM_Offset"                 , fBCM_Offset,                   kDouble,  (UInt_t) fNumBCMs},
      {"BCM_SatQuadratic"           , fBCM_SatQuadratic,             kDouble,  (UInt_t) fNumBCMs,1},
      {"BCM_SatOffset"              , fBCM_SatOffset,                kDouble,  (UInt_t) fNumBCMs,1},
      {"BCM_Current_threshold"      , &fbcm_Current_Threshold,       kDouble,  0 , 1},
      {"BCM_Current_threshold_index", &fbcm_Current_Threshold_Index, kInt   ,  0 , 1},
      {0}
    };
    fbcm_Current_Threshold = 0.0;
    fbcm_Current_Threshold_Index = 0;
    for(Int_t i=0;i<fNumBCMs;i++) {
      fBCM_SatOffset[i]=0.;
      fBCM_SatQuadratic[i]=0.;
    }
    err = LoadDB(file,date,list2,fPrefix); 
    // gHcParms->LoadParmValues((DBRequest*)&list2, prefix);
    string myStr;
    vector<string> bcm_names = Podd::vsplit(bcm_namelist);
    for(Int_t i=0;i<fNumBCMs;i++) {
       myStr = fName + ".bcm." + bcm_names[i] + ".current"; 
       fBCM_Name.push_back(myStr);
       fBCM_delta_charge[i]=0.;
    }
    // print what we have
    std::cout << "LOADED FROM DATABASE: " << std::endl; 
    for(Int_t i=0;i<fNumBCMs;i++){
       std::cout << Form("%s: offset = %.3lf Hz, gain = %.3lf Hz/uA",fBCM_Name[i].c_str(),fBCM_Offset[i],fBCM_Gain[i]) << std::endl; 
    }
  }
// #endif
  fTotalTime=0.;
  fPrevTotalTime=0.;
  fDeltaTime=-1.;
  //
  //
  return kOK;
}
void SBSScalerEvtHandler::SetDelayedType(int evtype) {
  /**
   * \brief Delay analysis of this event type to end.
   *
   * Final scaler events generated in readout list end routines may not
   * come in order in the data stream.  If the event type of a end routine
   * scaler event is set, then the event contents will be saved and analyzed
   * at the end of the analysis so that time ordering of scaler events is preserved.
   */
  fDelayedType = evtype;
}
  
Int_t SBSScalerEvtHandler::Analyze(THaEvData *evdata)
{
  Int_t lfirst=1;

  if(evdata->GetEvNum() > 0) {
    evNumber  = evdata->GetEvNum();
    evNumberR = evNumber;
  }
  if ( !IsMyEvent(evdata->GetEvType()) ) return -1;

  if (fDebugFile) {
    *fDebugFile << endl << "---------------------------------- "<<endl<<endl;
    *fDebugFile << "\nEnter SBSScalerEvtHandler  for fName = "<<fName<<endl;
    EvDump(evdata);
  }

  if (lfirst && !fScalerTree) {


    lfirst = 0; // Can't do this in Init for some reason

    TString sname1 = "TS";
    TString sname2 = sname1 + fName;
    TString sname3 = fName + "  Scaler Data";

    if (fDebugFile) {
      *fDebugFile << "\nAnalyze 1st time for fName = "<<fName<<endl;
      *fDebugFile << sname2 << "      " <<sname3<<endl;
    }

    fScalerTree = new TTree(sname2.Data(),sname3.Data());
    fScalerTree->SetAutoSave(200000000);

    TString name, tinfo;

    name = "evcount";
    tinfo = name + "/D";
    fScalerTree->Branch(name.Data(), &evcountR, tinfo.Data(), 4000);
 
    name = "evNumber";
    tinfo = name + "/D";
    fScalerTree->Branch(name.Data(), &evNumberR, tinfo.Data(), 4000);

    // create a branch for the physics event number
    fScalerTree->Branch("evnum",&fPhysicsEventNumber,"evnum/L");

    for (size_t i = 0; i < scalerloc.size(); i++) {
      name = scalerloc[i]->name;
      tinfo = name + "/D";
      fScalerTree->Branch(name.Data(), &dvars[i], tinfo.Data(), 4000);
    }

  }  // if (lfirst && !fScalerTree)

  // get the physics event number 
  fPhysicsEventNumber = evdata->GetEvNum();

  UInt_t *rdata = (UInt_t*) evdata->GetRawDataBuffer();

  if( (Int_t)evdata->GetEvType() == fDelayedType) { // Save this event for processing later
    UInt_t evlen = evdata->GetEvLength();
    
    UInt_t *datacopy = new UInt_t[evlen];
    fDelayedEvents.push_back(datacopy);
    memcpy(datacopy,rdata,evlen*sizeof(UInt_t));
    return 1;
  } else { 			// A normal event
    if (fDebugFile) *fDebugFile<<"\n\nSBSScalerEvtHandler :: Debugging event type "<<dec<<evdata->GetEvType()<< " event num = " << evdata->GetEvNum() << endl<<endl;
    Int_t ret;
    if((ret=AnalyzeBuffer(rdata,fOnlySyncEvents))) {
      if (fDebugFile) *fDebugFile << "scaler tree ptr  "<<fScalerTree<<endl;
      if (fScalerTree) fScalerTree->Fill();
      
      //fill histos here?
      double Time = -10;
      double clk_cnt = 0, clk_rate = 0, unser_rate = 0, u1_rate = 0, unew_rate = 0, dnew_rate = 0, d1_rate = 0, d3_rate = 0, d10_rate = 0;
      for (UInt_t i = 0; i < scalerloc.size(); i++) {
	TString name = scalerloc[i]->name; 
	
	//cout << name.Data() << endl;
	
	if(name.Contains("4MHz_CLK")){
	  if(name.Contains("Rate")){
	    clk_rate = dvars[i];
	  }else if(name.Contains("scaler") && !name.Contains("Cut")){
	    clk_cnt = dvars[i];
	  }
	}
	
	if(name.Contains("bcm")){
	  if(name.Contains("unser.rate"))unser_rate = dvars[i];
	  if(name.Contains("u1.rate"))u1_rate = dvars[i];
	  if(name.Contains("unew.rate"))unew_rate = dvars[i];
	  if(name.Contains("dnew.rate"))dnew_rate = dvars[i];
	  if(name.Contains("d1.rate"))d1_rate = dvars[i];
	  if(name.Contains("d3.rate"))d3_rate = dvars[i];
	  if(name.Contains("d10.rate"))d10_rate = dvars[i];
	}
	
      }
      Time = clk_cnt/clk_rate;
      
      if(fIunserVsTime!=NULL && Time>0) fIunserVsTime->Fill(Time, unser_rate);
      if(fIu1VsTime!=NULL    && Time>0) fIu1VsTime->Fill(Time, u1_rate);
      if(fIunewVsTime!=NULL  && Time>0) fIunewVsTime->Fill(Time, unew_rate);
      if(fIdnewVsTime!=NULL  && Time>0) fIdnewVsTime->Fill(Time, dnew_rate);
      if(fId1VsTime!=NULL    && Time>0) fId1VsTime->Fill(Time, d1_rate);
      if(fId3VsTime!=NULL    && Time>0) fId3VsTime->Fill(Time, d3_rate);
      if(fId10VsTime!=NULL   && Time>0) fId10VsTime->Fill(Time, d10_rate);
    }
    return ret;

  }

}
Int_t SBSScalerEvtHandler::AnalyzeBuffer(UInt_t* rdata, Bool_t onlysync)
{

  // Parse the data, load local data arrays.
  UInt_t *p = (UInt_t*) rdata;

  UInt_t *plast = p+*p;		// Index to last word in the bank

  ifound=0;
  while(p<plast) {
    p++;			  // point to header
    if (fDebugFile) {
      *fDebugFile << "Bank: " << hex << *p << dec << " len: " << *(p-1) << endl;
    }
    if((*p & 0xff00) == 0x1000) {	// Bank Containing banks
      p++;				// Now pointing to a bank in the bank
    } else if (((*p & 0xff00) == 0x100) && (*p != 0xC0000100)) {
      // Bank containing integers.  Look for scalers
      // This is either ROC bank containing integers or
      // a bank within a ROC containing data from modules of a single type
      // Look for scaler data
      // Assume that very first word is a scaler header
      // At any point in the bank where the word is not a matching
      // header, we stop.
      UInt_t tag = (*p>>16) & 0xffff;
      UInt_t num = (*p) & 0xff;
      UInt_t *pnext = p+*(p-1);	// Next bank
      p++;			// First data word

      // Skip over banks that can't contain scalers
      // If SetOnlyBanks(kTRUE) called, fRocSet will be empty
      // so only bank tags matching module types will be considered.
      if(fModuleSet.find(tag)!=fModuleSet.end()) {
	if(onlysync && num==0) {
	  ifound = 0;
	  return 0;
	}
      } else if (fRocSet.find(tag)==fRocSet.end()) {
	p = pnext;		// Fall through to end of the above else if
      }

      // Look for normalization scaler module first.
      if(fNormIdx >= 0) {
	UInt_t *psave = p;
	while(p < pnext) {
	  if(scalers[fNormIdx]->IsSlot(*p)) {
	    scalers[fNormIdx]->Decode(p);
	    ifound = 1;
	    break;
	  }
	  p += scalers[fNormIdx]->GetNumChan() + 1;
	}
	p = psave;
      }
      while(p < pnext) {
	Int_t nskip = 0;
	if(fDebugFile) {
	  *fDebugFile << "Scaler Header: " << hex << *p << dec;
	}
	for(size_t j=0; j<scalers.size(); j++) {
	  if(scalers[j]->IsSlot(*p)) {
	    nskip = scalers[j]->GetNumChan() + 1;
	    if((Int_t) j != fNormIdx) {
	      if(fDebugFile) {
		*fDebugFile << " found (" << j << ")  skip " << nskip << endl;
	      }
	      scalers[j]->Decode(p);
	      ifound = 1;
	    }
	    break;
	  }
	}
	if(nskip == 0) {
	  if(fDebugFile) {
	    *fDebugFile << endl;
	  }
	  break;	// Didn't find a matching header
	}
	p = p + nskip;
      }
      p = pnext;
    } else {
      p = p+*(p-1);		// Skip to next bank
    }
  }

  if (fDebugFile) {
    *fDebugFile << "Finished with decoding.  "<<endl;
    *fDebugFile << "   Found flag   =  "<<ifound<<endl;
  }

  // HMS has headers which are different from SOS, but both are
  // event type 0 and come here.  If you found no headers, return.

  if (!ifound) return 0;

  // The correspondance between dvars and the scaler and the channel
  // will be driven by a scaler.map file  -- later
  Double_t scal_current=0;
  UInt_t thisClock = scalers[fNormIdx]->GetData(fClockChan);
  if(thisClock < fLastClock) {	// Count clock scaler wrap arounds
    fClockOverflows++;
  }
  fTotalTime = (thisClock+(((Double_t) fClockOverflows)*kMaxUInt+fClockOverflows))/fClockFreq;
  fLastClock = thisClock;
  fDeltaTime= fTotalTime - fPrevTotalTime;
  if (fDeltaTime==0) {
    cout << " *******************   Severe Warning ****************************" << endl;
    cout << " In SBSScalerEvtHandler have found fDeltaTime is zero !!   " << endl;
      cout << " ******************* Alert DAQ experts ****************************" << endl;
  }
  fPrevTotalTime=fTotalTime;
  Int_t nscal=0;
  for (size_t i = 0; i < scalerloc.size(); i++)  {
    size_t ivar = scalerloc[i]->ivar;
    size_t idx = scalerloc[i]->index;
    size_t ichan = scalerloc[i]->ichan;
    if (evcount==0) {
      if (fDebugFile) *fDebugFile << "Debug dvarsFirst "<<i<<"   "<<ivar<<"  "<<idx<<"  "<<ichan<<endl;
      if ((ivar < scalerloc.size()) &&
	  (idx < scalers.size()) &&
	  (ichan < MAXCHAN)){
	if(fUseFirstEvent){
	  if (scalerloc[ivar]->ikind == ICOUNT){
	    UInt_t scaldata = scalers[idx]->GetData(ichan);
	    dvars[ivar] = scaldata;
	    scal_present_read.push_back(scaldata);
	    scal_prev_read.push_back(0);
	    scal_overflows.push_back(0);
	    dvarsFirst[ivar] = 0.0;
	  }
	  if (scalerloc[ivar]->ikind == ITIME){
	    dvars[ivar] =fTotalTime;
	    dvarsFirst[ivar] = 0;
	  }
	  if (scalerloc[ivar]->ikind == IRATE) {
	    dvars[ivar] = (scalers[idx]->GetData(ichan))/fDeltaTime;
	    dvarsFirst[ivar] = dvars[ivar];            
	    //printf("%s %f\n",scalerloc[ivar]->name.Data(),scalers[idx]->GetRate(ichan)); //checks
	  }
	  if(scalerloc[ivar]->ikind == ICURRENT || scalerloc[ivar]->ikind == ICHARGE){
	    Int_t bcm_ind=-1;
	    for(Int_t itemp =0; itemp<fNumBCMs;itemp++)
	      {		
		size_t match = string(scalerloc[ivar]->name.Data()).find(string(fBCM_Name[itemp]));
		if (match!=string::npos)
		  {
		    bcm_ind=itemp;
		  }
	      }
	    if (scalerloc[ivar]->ikind == ICURRENT) {
              dvars[ivar]=0.;
	      if (bcm_ind != -1) {
                 dvars[ivar]=((scalers[idx]->GetData(ichan))/fDeltaTime-fBCM_Offset[bcm_ind])/fBCM_Gain[bcm_ind];
		 dvars[ivar]=dvars[ivar]+fBCM_SatQuadratic[bcm_ind]*TMath::Power(TMath::Max(dvars[ivar]-fBCM_SatOffset[bcm_ind],0.0),2.0);

	      }
         	if (bcm_ind == fbcm_Current_Threshold_Index) scal_current= dvars[ivar];
	    }
	    if (scalerloc[ivar]->ikind == ICHARGE) {
	      if (bcm_ind != -1) {
		Double_t cur_temp=((scalers[idx]->GetData(ichan))/fDeltaTime-fBCM_Offset[bcm_ind])/fBCM_Gain[bcm_ind];
		cur_temp=cur_temp+fBCM_SatQuadratic[bcm_ind]*TMath::Power(TMath::Max(cur_temp-fBCM_SatOffset[bcm_ind],0.0),2.0);
		fBCM_delta_charge[bcm_ind]=fDeltaTime*cur_temp;
		dvars[ivar]+=fBCM_delta_charge[bcm_ind];
	      }
	    }
	    //	    printf("1st event %i index %i fBCMname %s scalerloc %s offset %f gain %f computed %f\n",evcount, bcm_ind, fBCM_Name[bcm_ind],scalerloc[ivar]->name.Data(),fBCM_Offset[bcm_ind],fBCM_Gain[bcm_ind],dvars[ivar]);
	  }
	  
	  if (fDebugFile) *fDebugFile << "   dvarsFirst  "<<scalerloc[ivar]->ikind<<"  "<<dvarsFirst[ivar]<<endl;
	  
	} else { //ifnotusefirstevent
	  if (scalerloc[ivar]->ikind == ICOUNT) {
              dvarsFirst[ivar] = scalers[idx]->GetData(ichan) ;
              scal_present_read.push_back(dvarsFirst[ivar]);
              scal_prev_read.push_back(0);
	  }
	  if (scalerloc[ivar]->ikind == ITIME){
	    dvarsFirst[ivar] = fTotalTime;
	  }
	  if (scalerloc[ivar]->ikind == IRATE)  {
	    dvarsFirst[ivar] = (scalers[idx]->GetData(ichan))/fDeltaTime;
 	    //printf("%s %f\n",scalerloc[ivar]->name.Data(),scalers[idx]->GetRate(ichan)); //checks
	  }
	  if(scalerloc[ivar]->ikind == ICURRENT || scalerloc[ivar]->ikind == ICHARGE)
	    {
	      Int_t bcm_ind=-1;
	      for(Int_t itemp =0; itemp<fNumBCMs;itemp++)
		{		
		  size_t match = string(scalerloc[ivar]->name.Data()).find(string(fBCM_Name[itemp]));
		  if (match!=string::npos)
		    {
		      bcm_ind=itemp;
		    }
		}
	    if (scalerloc[ivar]->ikind == ICURRENT) {
	        dvarsFirst[ivar]=0.0;
                if (bcm_ind != -1) {
                 dvarsFirst[ivar]=((scalers[idx]->GetData(ichan))/fDeltaTime-fBCM_Offset[bcm_ind])/fBCM_Gain[bcm_ind];
		 dvarsFirst[ivar]=dvarsFirst[ivar]+fBCM_SatQuadratic[bcm_ind]*TMath::Power(TMath::Max(dvars[ivar]-fBCM_SatOffset[bcm_ind],0.0),2.);
		}
         	if (bcm_ind == fbcm_Current_Threshold_Index) scal_current= dvarsFirst[ivar];
	    }
	    if (scalerloc[ivar]->ikind == ICHARGE) {
	      if (bcm_ind != -1) {
		Double_t cur_temp=((scalers[idx]->GetData(ichan))/fDeltaTime-fBCM_Offset[bcm_ind])/fBCM_Gain[bcm_ind];
		cur_temp=cur_temp+fBCM_SatQuadratic[bcm_ind]*TMath::Power(TMath::Max(cur_temp-fBCM_SatOffset[bcm_ind],0.0),2.);
		fBCM_delta_charge[bcm_ind]=fDeltaTime*cur_temp;
               dvarsFirst[ivar]+=fBCM_delta_charge[bcm_ind];
	      }
	    }
	    }
	  if (fDebugFile) *fDebugFile << "   dvarsFirst  "<<scalerloc[ivar]->ikind<<"  "<<dvarsFirst[ivar]<<endl;
	}
      } 
      else {
	cout << "SBSScalerEvtHandler:: ERROR:: incorrect index "<<ivar<<"  "<<idx<<"  "<<ichan<<endl;
      }
    }else{ // evcount != 0
      if (fDebugFile) *fDebugFile << "Debug dvars "<<i<<"   "<<ivar<<"  "<<idx<<"  "<<ichan<<endl;
      if ((ivar < scalerloc.size()) &&
	  (idx < scalers.size()) &&
	  (ichan < MAXCHAN)) {
	if (scalerloc[ivar]->ikind == ICOUNT) {
	    UInt_t scaldata = scalers[idx]->GetData(ichan);
	    if(scaldata < scal_prev_read[nscal]) {
	      scal_overflows[nscal]++;
	    }
            dvars[ivar] = scaldata + (1+((Double_t)kMaxUInt))*scal_overflows[nscal]
	      -dvarsFirst[ivar];
            scal_present_read[nscal]=scaldata;
	    nscal++;
	}
	if (scalerloc[ivar]->ikind == ITIME) {
	  dvars[ivar] = fTotalTime;
	}
	if (scalerloc[ivar]->ikind == IRATE) {
	  UInt_t scaldata = scalers[idx]->GetData(ichan);
	  UInt_t diff;
	  if(scaldata < scal_prev_read[nscal-1]) {
	    diff = (kMaxUInt-(scal_prev_read[nscal-1] - 1)) + scaldata;
	  } else {
	    diff = scaldata - scal_prev_read[nscal-1];
	  }
	  dvars[ivar] =  diff/fDeltaTime;
	  // printf("%s %f\n",scalerloc[ivar]->name.Data(),scalers[idx]->GetRate(ichan));//checks
	}
	if(scalerloc[ivar]->ikind == ICURRENT || scalerloc[ivar]->ikind == ICHARGE)
	  {
	    Int_t bcm_ind=-1;
	    for(Int_t itemp =0; itemp<fNumBCMs;itemp++)
	      {		
		size_t match = string(scalerloc[ivar]->name.Data()).find(string(fBCM_Name[itemp]));
		if (match!=string::npos)
		  {
		    bcm_ind=itemp;
		  }
	      }
	    if (scalerloc[ivar]->ikind == ICURRENT) {
              dvars[ivar]=0;
	      if (bcm_ind != -1) {
		UInt_t scaldata = scalers[idx]->GetData(ichan);
		UInt_t diff;
		if(scaldata < scal_prev_read[nscal-1]) {
		  diff = (kMaxUInt-(scal_prev_read[nscal-1] - 1)) + scaldata;
		} else {
		  diff = scaldata - scal_prev_read[nscal-1];
		}
		dvars[ivar]=0.;
 		if (fDeltaTime>0) {
		Double_t cur_temp=(diff/fDeltaTime-fBCM_Offset[bcm_ind])/fBCM_Gain[bcm_ind];
		cur_temp=cur_temp+fBCM_SatQuadratic[bcm_ind]*TMath::Power(TMath::Max(cur_temp-fBCM_SatOffset[bcm_ind],0.0),2.);
		  
		dvars[ivar]=cur_temp;
		}
	      }
	      if (bcm_ind == fbcm_Current_Threshold_Index) scal_current= dvars[ivar];
	    }
	    if (scalerloc[ivar]->ikind == ICHARGE) {
	      if (bcm_ind != -1) {
		UInt_t scaldata = scalers[idx]->GetData(ichan);
		UInt_t diff;
		if(scaldata < scal_prev_read[nscal-1]) {
		  diff = (kMaxUInt-(scal_prev_read[nscal-1] - 1)) + scaldata;
		} else {
		  diff = scaldata - scal_prev_read[nscal-1];
		}
		fBCM_delta_charge[bcm_ind]=0;
		if (fDeltaTime>0)  {
		  Double_t cur_temp=(diff/fDeltaTime-fBCM_Offset[bcm_ind])/fBCM_Gain[bcm_ind];
		  cur_temp=cur_temp+fBCM_SatQuadratic[bcm_ind]*TMath::Power(TMath::Max(cur_temp-fBCM_SatOffset[bcm_ind],0.0),2.);
		fBCM_delta_charge[bcm_ind]=fDeltaTime*cur_temp;
		}
                 dvars[ivar]+=fBCM_delta_charge[bcm_ind];
	      }
	    }
	    //	    printf("event %i index %i fBCMname %s scalerloc %s offset %f gain %f computed %f\n",evcount, bcm_ind, fBCM_Name[bcm_ind],scalerloc[ivar]->name.Data(),fBCM_Offset[bcm_ind],fBCM_Gain[bcm_ind],dvars[ivar]);
	  }
	if (fDebugFile) *fDebugFile << "   dvars  "<<scalerloc[ivar]->ikind<<"  "<<dvars[ivar]<<endl;
      } else {
	cout << "SBSScalerEvtHandler:: ERROR:: incorrect index "<<ivar<<"  "<<idx<<"  "<<ichan<<endl;
      }
    }
    
  }
  //
  for (size_t i = 0; i < scalerloc.size(); i++)  {
    size_t ivar = scalerloc[i]->ivar;
    size_t idx = scalerloc[i]->index;
    size_t ichan = scalerloc[i]->ichan;
    if (scalerloc[ivar]->ikind == ICUT+ICOUNT){
      UInt_t scaldata = scalers[idx]->GetData(ichan);
      if ( scal_current > fbcm_Current_Threshold) {
	UInt_t diff;
	if(scaldata < dvars_prev_read[ivar]) {
	  diff = (kMaxUInt-(dvars_prev_read[ivar] - 1)) + scaldata;
	} else {
	  diff = scaldata - dvars_prev_read[ivar];
	}
	dvars[ivar] += diff;
      } 
      dvars_prev_read[ivar] = scaldata;
    }
    if (scalerloc[ivar]->ikind == ICUT+ICHARGE){
	    Int_t bcm_ind=-1;
	    for(Int_t itemp =0; itemp<fNumBCMs;itemp++)
	      {		
		size_t match = string(scalerloc[ivar]->name.Data()).find(string(fBCM_Name[itemp]));
		if (match!=string::npos)
		  {
		    bcm_ind=itemp;
		  }
	      }
      if ( scal_current > fbcm_Current_Threshold && bcm_ind != -1) {
	dvars[ivar] += fBCM_delta_charge[bcm_ind];
     } 
    }
    if (scalerloc[ivar]->ikind == ICUT+ITIME){
      if ( scal_current > fbcm_Current_Threshold) {
         dvars[ivar] += fDeltaTime;
      } 
    }
  }
  //
  evcount = evcount + 1;
  evcountR = evcount;
  //
  for (size_t j=0; j<scal_prev_read.size(); j++) scal_prev_read[j]=scal_present_read[j];
  //  
  for (auto & scaler : scalers) scaler->Clear();
  
  return 1;
}


THaAnalysisObject::EStatus SBSScalerEvtHandler::Init(const TDatime& date)
{
  //
  ReadDatabase(date);
  const int LEN = 200;
  char cbuf[LEN];
 
  eventtypes.push_back(1);  

  fStatus = kOK;
  fNormIdx = -1;

  for( vector<UInt_t*>::iterator it = fDelayedEvents.begin();
       it != fDelayedEvents.end(); ++it )
    delete [] *it;
  fDelayedEvents.clear();

  cout << "Initializing SBSScalerEvtHandler; name = "
        << fName << endl;

  if(eventtypes.size()==0) {
    eventtypes.push_back(0);  // Default Event Type
  }

  TString dfile;
  dfile = fName + "scaler.txt";

// Parse the map file which defines what scalers exist and the global variables.

  TString sname0 = "Scalevt";
  TString sname;
  sname = fName+sname0;

  FILE *fi = Podd::OpenDBFile(sname.Data(), date);
  if ( !fi ) {
    cout << "Cannot find db file for "<<fName<<" scaler event handler"<<endl;
    return kFileError;
  }

  size_t minus1 = -1;
  size_t pos1;
  string scomment = "#";
  string svariable = "variable";
  string smap = "map";
  vector<string> dbline;

  while( fgets(cbuf, LEN, fi) != NULL) {
    std::string sin(cbuf);
    std::string sinput(sin.substr(0,sin.find_first_of("#")));
    if (fDebugFile) *fDebugFile << "string input "<<sinput<<endl;
    dbline = Podd::vsplit(sinput);
    if (dbline.size() > 0) {
      pos1 = FindNoCase(dbline[0],scomment);
      if (pos1 != minus1) continue;
      pos1 = FindNoCase(dbline[0],svariable);
      if (pos1 != minus1 && dbline.size()>4) {
	string sdesc = "";
	for (size_t j=5; j<dbline.size(); j++) sdesc = sdesc+" "+dbline[j];
	UInt_t islot = atoi(dbline[1].c_str());
	UInt_t ichan = atoi(dbline[2].c_str());
	UInt_t ikind = atoi(dbline[3].c_str());
       	if (fDebugFile)
	  *fDebugFile << "add var "<<dbline[1]<<"   desc = "<<sdesc<<"    islot= "<<islot<<"  "<<ichan<<"  "<<ikind<<endl;
	TString tsname(dbline[4].c_str());
	TString tsdesc(sdesc.c_str());
	AddVars(tsname,tsdesc,islot,ichan,ikind);
	// add extra scaler which is cut on the current
	if (ikind == ICOUNT ||ikind == ITIME ||ikind == ICHARGE  ) {
	  tsname=tsname+"Cut";
	  AddVars(tsname,tsdesc,islot,ichan,ICUT+ikind);
	  }
      }
      pos1 = FindNoCase(dbline[0],smap);
      if (fDebugFile) *fDebugFile << "map ? "<<dbline[0]<<"  "<<smap<<"   "<<pos1<<"   "<<dbline.size()<<endl;
      if (pos1 != minus1 && dbline.size()>6) {
	Int_t imodel, icrate, islot, inorm;
	UInt_t header, mask;
	char cdum[20];
	sscanf(sinput.c_str(),"%s %d %d %d %x %x %d \n",cdum,&imodel,&icrate,&islot, &header, &mask, &inorm);
	if ((fNormSlot >= 0) && (fNormSlot != inorm)) cout << "SBSScalerEvtHandler::WARN:  contradictory norm slot  "<<fNormSlot<<"   "<<inorm<<endl;
	fNormSlot = inorm;  // slot number used for normalization.  This variable is not used but is checked.
	Int_t clkchan = -1;
	Double_t clkfreq = 1;
	if (dbline.size()>8) {
	  clkchan = atoi(dbline[7].c_str());
	  clkfreq = 1.0*atoi(dbline[8].c_str());
	  fClockChan=clkchan;
	  fClockFreq=clkfreq;
	}
	if (fDebugFile) {
	  *fDebugFile << "map line "<<dec<<imodel<<"  "<<icrate<<"  "<<islot<<endl;
	  *fDebugFile <<"   header  0x"<<hex<<header<<"  0x"<<mask<<dec<<"  "<<inorm<<"  "<<clkchan<<"  "<<clkfreq<<endl;
	}
	switch (imodel) {
	case 560:
	  scalers.push_back(new Scaler560(icrate, islot));
	  if(!fOnlyBanks) fRocSet.insert(icrate);
	  fModuleSet.insert(imodel);
	  break;
	case 1151:
	  scalers.push_back(new Scaler1151(icrate, islot));
	  if(!fOnlyBanks) fRocSet.insert(icrate);
	  fModuleSet.insert(imodel);
	  break;
	case 3800:
	  scalers.push_back(new Scaler3800(icrate, islot));
	  if(!fOnlyBanks) fRocSet.insert(icrate);
	  fModuleSet.insert(imodel);
	  break;
	case 3801:
	  scalers.push_back(new Scaler3801(icrate, islot));
	  if(!fOnlyBanks) fRocSet.insert(icrate);
	  fModuleSet.insert(imodel);
	  break;
	  //	case 9001:		// TI Scalers
	  //	  scalers.push_back(new Scaler9001(icrate, islot));
	  //	  if(!fOnlyBanks) fRocSet.insert(icrate);
	  //	  fModuleSet.insert(imodel);
	  //	  break;
	  //	case 9250:		// FADC250 Scalers
	  //	  scalers.push_back(new Scaler9250(icrate, islot));
	  //	  if(!fOnlyBanks) fRocSet.insert(icrate);
	  //	  fModuleSet.insert(imodel);
	  //	  break;
	}
	if (scalers.size() > 0) {
	  UInt_t idx = scalers.size()-1;
	  // Headers must be unique over whole event, not
	  // just within a ROC
	  scalers[idx]->SetHeader(header, mask);
// The normalization slot has the clock in it, so we automatically recognize it.
// fNormIdx is the index in scaler[] and 
// fNormSlot is the slot#, checked for consistency
	  if (clkchan >= 0) {
		  scalers[idx]->SetClock(defaultDT, clkchan, clkfreq);
		  cout << "Setting scaler clock ... channel = "<<clkchan<<" ... freq = "<<clkfreq<<endl;
		  if (fDebugFile) *fDebugFile <<"Setting scaler clock ... channel = "<<clkchan<<" ... freq = "<<clkfreq<<endl;
		  fNormIdx = idx;
		  if (islot != fNormSlot) cout << "SBSScalerEvtHandler:: WARN: contradictory norm slot ! "<<islot<<endl;  
		  
	  }
	}
      }
    }
  }
  // can't compare UInt_t to Int_t (compiler warning), so do this
  nscalers=0;
  for (size_t i=0; i<scalers.size(); i++) nscalers++;
  // need to do LoadNormScaler after scalers created and if fNormIdx found
  if (fDebugFile) *fDebugFile <<"fNormIdx = "<<fNormIdx<<endl;
  if ((fNormIdx >= 0) && fNormIdx < nscalers) {
    for (Int_t i = 0; i < nscalers; i++) {
      if (i==fNormIdx) continue;
      scalers[i]->LoadNormScaler(scalers[fNormIdx]);
    }
  }

#ifdef HARDCODED
  // This code is superseded by the parsing of a map file above.  It's another way ...
  if (fName == "Left") {
    AddVars("TSbcmu1", "BCM x1 counts", 1, 4, ICOUNT);
    AddVars("TSbcmu1r","BCM x1 rate",  1, 4, IRATE);
    AddVars("TSbcmu3", "BCM u3 counts", 1, 5, ICOUNT);
    AddVars("TSbcmu3r", "BCM u3 rate",  1, 5, IRATE);
  } else {
    AddVars("TSbcmu1", "BCM x1 counts", 0, 4, ICOUNT);
    AddVars("TSbcmu1r","BCM x1 rate",  0, 4, IRATE);
    AddVars("TSbcmu3", "BCM u3 counts", 0, 5, ICOUNT);
    AddVars("TSbcmu3r", "BCM u3 rate",  0, 5, IRATE);
  }
#endif


  DefVars();

#ifdef HARDCODED
  // This code is superseded by the parsing of a map file above.  It's another way ...
  if (fName == "Left") {
    scalers.push_back(new Scaler1151(1,0));
    scalers.push_back(new Scaler3800(1,1));
    scalers.push_back(new Scaler3800(1,2));
    scalers.push_back(new Scaler3800(1,3));
    scalers[0]->SetHeader(0xabc00000, 0xffff0000);
    scalers[1]->SetHeader(0xabc10000, 0xffff0000);
    scalers[2]->SetHeader(0xabc20000, 0xffff0000);
    scalers[3]->SetHeader(0xabc30000, 0xffff0000);
    scalers[0]->LoadNormScaler(scalers[1]);
    scalers[1]->SetClock(4, 7, 1024);
    scalers[2]->LoadNormScaler(scalers[1]);
    scalers[3]->LoadNormScaler(scalers[1]);
  } else {
    scalers.push_back(new Scaler3800(2,0));
    scalers.push_back(new Scaler3800(2,0));
    scalers.push_back(new Scaler1151(2,1));
    scalers.push_back(new Scaler1151(2,2));
    scalers[0]->SetHeader(0xceb00000, 0xffff0000);
    scalers[1]->SetHeader(0xceb10000, 0xffff0000);
    scalers[2]->SetHeader(0xceb20000, 0xffff0000);
    scalers[3]->SetHeader(0xceb30000, 0xffff0000);
    scalers[0]->SetClock(4, 7, 1024);
    scalers[1]->LoadNormScaler(scalers[0]);
    scalers[2]->LoadNormScaler(scalers[0]);
    scalers[3]->LoadNormScaler(scalers[0]);
  }
#endif

  // Verify that the slots are not defined twice
  for (UInt_t i1=0; i1 < scalers.size()-1; i1++) {
    for (UInt_t i2=i1+1; i2 < scalers.size(); i2++) {
      if (scalers[i1]->GetSlot()==scalers[i2]->GetSlot())
	cout << "SBSScalerEvtHandler:: WARN:  same slot defined twice"<<endl;
    }
  }
  // Identify indices of scalers[] vector to variables.
  for (UInt_t i=0; i < scalers.size(); i++) {
    for (UInt_t j = 0; j < scalerloc.size(); j++) {
      if (scalerloc[j]->islot==static_cast<UInt_t>(scalers[i]->GetSlot()))
	scalerloc[j]->index = i;
    }
  }

  if(fDebugFile) *fDebugFile << "SBSScalerEvtHandler:: Name of scaler bank "<<fName<<endl;
  for (size_t i=0; i<scalers.size(); i++) {
    if(fDebugFile) {
      *fDebugFile << "Scaler  #  "<<i<<endl;
      scalers[i]->SetDebugFile(fDebugFile);
      scalers[i]->DebugPrint(fDebugFile);
    }
  }

 
  //
  return kOK;
}

void SBSScalerEvtHandler::AddVars(TString name, TString desc, UInt_t islot,
				  UInt_t ichan, UInt_t ikind)
{
  // need to add fName here to make it a unique variable.  (Left vs Right HRS, for example)
  TString name1 = fName + name;
  TString desc1 = fName + desc;
  // We don't yet know the correspondence between index of scalers[] and slots.
  // Will put that in later.
  scalerloc.push_back( new HCScalerLoc(name1, desc1, 0, islot, ichan, ikind,
				       scalerloc.size()) );
}

void SBSScalerEvtHandler::DefVars()
{
  // called after AddVars has finished being called.
  Nvars = scalerloc.size();
  if (Nvars == 0) return;
  dvars_prev_read = new UInt_t[Nvars];  // dvars_prev_read is a member of this class
  dvars = new Double_t[Nvars];  // dvars is a member of this class
  dvarsFirst = new Double_t[Nvars];  // dvarsFirst is a member of this class
  memset(dvars, 0, Nvars*sizeof(Double_t));
  memset(dvars_prev_read, 0, Nvars*sizeof(UInt_t));
  memset(dvarsFirst, 0, Nvars*sizeof(Double_t));
  if (gHaVars) {
    if(fDebugFile) *fDebugFile << "SBSScalerEVtHandler:: Have gHaVars "<<gHaVars<<endl;
  } else {
    cout << "No gHaVars ?!  Well, that's a problem !!"<<endl;
    return;
  }
  if(fDebugFile) *fDebugFile << "SBSScalerEvtHandler:: scalerloc size "<<scalerloc.size()<<endl;
  const Int_t* count = 0;
  for (size_t i = 0; i < scalerloc.size(); i++) {
    gHaVars->DefineByType(scalerloc[i]->name.Data(), scalerloc[i]->description.Data(),
			  &dvars[i], kDouble, count);
    //gHaVars->DefineByType(scalerloc[i]->name.Data(), scalerloc[i]->description.Data(),
    //			  &dvarsFirst[i], kDouble, count);
  }
}

size_t SBSScalerEvtHandler::FindNoCase(const string& sdata, const string& skey)
{
  // Find iterator of word "sdata" where "skey" starts.  Case insensitive.
  string sdatalc, skeylc;
  sdatalc = "";  skeylc = "";
  for (string::const_iterator p =
	 sdata.begin(); p != sdata.end(); ++p) {
    sdatalc += tolower(*p);
  }
  for (string::const_iterator p =
	 skey.begin(); p != skey.end(); ++p) {
    skeylc += tolower(*p);
  }
  if (sdatalc.find(skeylc,0) == string::npos) return -1;
  return sdatalc.find(skeylc,0);
};

ClassImp(SBSScalerEvtHandler)

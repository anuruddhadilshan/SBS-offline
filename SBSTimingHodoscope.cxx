//////////////////////////////////////////////////////////////////////////
//                                                                          
// SBSTimingHodoscope                                                            
//                                                                          
// General F1TDC supported scintilator plane detector
//                                                                          
//////////////////////////////////////////////////////////////////////////
//	
//	Author : Copy from AGen Lib neutron detection
//	Modify History:
//		Jin Huang <mailto:jinhuang@jlab.org>    July 2007	
//			make database file supporting comment
//			disable Multi Hit which is for neutron detection
//      Jin Huang    Mar 2007
//          Threshold of error reference per event that will pop up warnings
//      Jin Huang    Sept 2008
//          Convert to a detector version of SBSTimingHodoscope
//		Jin Huang	Mar 2009
//			Add fMaxADCHitBar; //Hit Bar number with max sqrt(RApedc*LApedc)
//				fMaxEnergyHitBar; //Hit Bar number with max sqrt(RE*LE)
//		Jin Huang	Mar 2009
//			Add flag   fCoarseProcessed & fFineProcessed
//			to help carry out CoarseProcess before CoarseTracking
//
//////////////////////////////////////////////////////////////////////////

#include <cstring>
#include <cstdio>
#include <iostream>
#include <sstream>

//#include "THaNeutronDetector.h"
//#include "THaMultiHit.h"

#include "THaEvData.h"
#include "THaDetMap.h"
#include "VarDef.h"
#include "VarType.h"
#include "TClonesArray.h"
#include "TMath.h"
#include "THaNonTrackingDetector.h"
#include <TTree.h>
#include <cmath>
//#include "THaDB.h"
#include "TVector3.h"
#include "THaOutput.h"

#include "SBSScintHit.h"
#include "SBSScintBar.h"
#include "SBSScintPMT.h"
#include "SBSAdcHit.h"
#include "SBSTdcHit.h"
#include "SBSScintPartialHit.h"
#include "THaTrack.h"
#include "THaTrackProj.h"

//put this header file below all other headers
#include "SBSTimingHodoscope.h"


using namespace std;

//////////////////////////////////////////////////////////////////////////
#if DEBUG_LEVEL>=4//massive info
#define DEBUG_THIS 1
#endif
//////////////////////////////////////////////////////////////////////////

//#define GO_OVER_COMMENT(file,cbuff,LEN) while ( ReadNumberSignStartComment( file, cbuff, LEN ));	//jump throug comment lines

ClassImp(SBSTimingHodoscope)

//____________________________________________________________________________
SBSTimingHodoscope::SBSTimingHodoscope( const char* name, const char* description,
								   THaApparatus* apparatus) :
THaNonTrackingDetector(name,description,apparatus)
{


	DEBUG_LEVEL_RELATED_PERFORMACE_CHECKER;

	DEBUG_HALL_A_ANALYZER_DEBUGER_INIT;



	// Persistent properties of the class
	fBars  = new TClonesArray("SBSScintBar", 100 );
	fRefCh       = new TClonesArray("SBSScintPMT", 5 );

	// per-event information
	fHits   = new TClonesArray("SBSScintHit", 200);
	//fCombHits   = new TClonesArray("THaMultiHit", 200);

	fLaHits = new TClonesArray("SBSAdcHit", 200);
	fRaHits = new TClonesArray("SBSAdcHit", 200);
	fLtHits = new TClonesArray("SBSTdcHit", 200);
	fRtHits = new TClonesArray("SBSTdcHit", 200);
	fPartHits = new TClonesArray("SBSScintPartialHit", 200);

	fTrackProj = new TClonesArray( "THaTrackProj", 5 );
	fRefHits     = new TClonesArray("SBSTdcHit", 5 );

	//fThreshold = 0.;
	fNBars = 0;
	fLE = NULL;
	fRE = NULL;

	fLrawA = NULL;
	fRrawA = NULL;
	fLpedcA = NULL;
	fRpedcA = NULL;

	fLT = NULL;
	fRT = NULL;
	fLTcounter = NULL;
	fRTcounter = NULL;
	hitcounter = NULL;
	Energy = NULL;		
	TDIFF = NULL;
	TOF = NULL;
	T_tot = NULL;
	Yt_pos = NULL;
	Ya_pos = NULL;
	Y_pred = NULL;
	Y_dev = NULL;
	fAngle = NULL;

	fLtIndex = NULL;
	fRtIndex = NULL;
	fLaIndex = NULL;
	fRaIndex = NULL;

	fEventCount=0;
	fErrorReferenceChCount=0;
	fErrorReferenceChRateWarningThreshold=1;
	fTooManyErrRefCh=false;

	fMatchRatioTrack=0;
}
//_____________________________________________________________________________

SBSTimingHodoscope::SBSTimingHodoscope( ) : THaNonTrackingDetector("plane","scintplane",0),
fBars(0), fNBars(0), fHits(0), /*fCombHits(0), */
fRefHits(0),
fLaHits(0), fRaHits(0),
fLtHits(0), fRtHits(0),
fPartHits(0), fRefCh(0),//fThreshold(0),
fLE(0), fRE(0), fLrawA(0), fRrawA(0),
fLpedcA(0), fRpedcA(0), fLT(0), fRT(0),
fLTcounter(0), fRTcounter(0), hitcounter(0),
Energy(0), TDIFF(0), TOF(0), T_tot(0),
Yt_pos(0), Ya_pos(0), Y_pred(0), Y_dev(0),
fAngle(0)
{
	// default constructors for input/output
	fLtIndex = NULL;
	fRtIndex = NULL;
	fLaIndex = NULL;
	fRaIndex = NULL;
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::GetParameter( FILE* file, const TString tag, Double_t* value  )
{  
	static const char* const here = "ReadDatabase::GetParameter";
	const int LEN = 200;
	char buff[LEN];
	char cbuff[LEN];	//commment buffer

	if( !file ) return kFileError;

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

	if (fgets (buff, LEN, file) == NULL ) {
		Error(Here(here), "Unexpected end of database file" );
		fclose(file);
		return kFileError;
	}
	TString line = buff;
	if ( line.BeginsWith(tag) ) {
		line.Replace(0,tag.Length(),"");
		line.Append("\0");
		sscanf(line.Data(),"%lf", value); 
		return 0 ; 
	} else {
		Error(Here(here), "Database file corrupted" );
		return kFileError;
	}
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::GetTable( FILE* file, const TString tag, Double_t* value , const Int_t maxval, int* first, int* last )
{  
	static const char* const here = "ReadDatabase::GetTable";
	const int LEN = 200;
	char buff[LEN];
	char cbuff[LEN];	//commment buffer
	Bool_t found = false;

	if( !file ) return kFileError;

	while (!found) {

		if (fgets (buff, LEN, file) == NULL) {
			return -1;
		}
		char* buf = ::Compress(buff);  //strip blanks
		TString line = buf;
		delete [] buf;
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
		line.ToLower();
		if ( tag == line ) 
			found = true;
	}

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

	fgets( buff, LEN, file );
	if( sscanf( buff, "%d%d",first,last) != 2 ) {
		if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
		Error( Here(here), "Error 1 reading %s data line: %s", tag.Data() , buff );
		return kInitError;
	}

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));	


	if (((*first)>=0)&&((*first)<maxval)&&((*last)>=(*first))&&((*last)<maxval)&&((*last)-(*first)+1<=maxval)) {
		Int_t i=0;
		Double_t val;
		fpos_t fpos;
		fgetpos(file,&fpos);
		while ((fscanf(file, "%lg",&val)==1)&&(i<maxval)) {
			value[i]=val;
			i++;
			fgetpos(file,&fpos);
		}
		if (i!=(*last)-(*first)+1) {
			fsetpos(file,&fpos);
			fgets(buff, LEN, file);
			Error( Here(here), "Error 2 reading %s data: check the number of entries. Last saw line: %s", tag.Data(), buff );
			return kInitError;
		}
	} else {
		Error( Here(here), "Error 3 reading %s data line: %s", tag.Data(), buff );
		return kInitError;
	}

#if DEBUG_LEVEL>=3
	Info(Here("GetTable"),"Print Table %s [%d]->[%d]: "
		,tag.Data(),(*first),(*last));
	for (Int_t i=(*first);i<=(*last);i++)
	{
		Printf("\t%f",value[i]);
	}
#endif

	return 0;
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::ReadDatabase(const TDatime& date)
{
  // Read this detector's parameters from the database file 'fi'.
  // This function is called by THaDetectorBase::Init() once at the
  // beginning of the analysis.
  // 'date' contains the date/time of the run being analyzed.

  // We can use this name here for logs
  static const char* const here = "ReadDatabase()";

  FILE* file = OpenFile( date );
  if( !file ) return kFileError;

  // Read in required geometry variables, which include fOrigin and fSize and 
  Int_t err = ReadGeometry( file, date, true );
  if( err ) {
    fclose(file);
    return err;
  }

  // Some temporary variables which we'll use to read in the database
  std::vector<Int_t> detmap_adc;
  std::vector<Int_t> detmap, chanmap;
  Int_t nbars,nref_chans = 0;
  std::vector<Float_t> xyz, dxyz;
  std::vector<Float_t> hit_acceptance;
  std::vector<Float_t> ref_ch_res;

  // Read mapping/geometry/configuration parameters
  fChanMapStart = 0;
  DBRequest config_request[] = {
    { "adc_detmap",   &detmap_adc,  kIntV, 0, true }, ///< Detector map for ADCs [optional]
    { "detmap",       &detmap,  kIntV }, ///< Detector map
    { "chanmap",      &chanmap, kIntV,    0, true }, ///< Optional channel map
    { "nbars",     &nbars,   kInt, 1, true }, ///< Number of modules
    { "dxdydz",       &dxyz,     kFloatV, 3 },  ///< block spacing (dx,dy,dz)
    { "hit_acceptance", &hit_acceptance, kFloatV, 2}, ///< fHitAcceptanceDx and Dy
    { "ref_ch_res", &ref_ch_res, kFloatV}, ///< Reference channel resolution
    { 0 } ///< Request must end in a NULL
  };

  // Clear out the old channel map before reading a new one
  fChanMap.clear();
  err = LoadDB( file, date, config_request, fPrefix );
  if(err) return err; // Return on any errors already

  // Sanity checks (make sure there were no inconsistent values entered)
  if(nbars <= 0) {
    Error( Here(here), "Must have at least one bar, number of bars specified: %d",
        nbars);
    return kInitError;  // Error already printed by FillDetMap
  }
  Int_t nlpmt,nrpmt;
  nlpmt = nrpmt = nbars;
  Int_t npmt = nlpmt+nrpmt;

  // Clear out the old channel map before reading a new one
  fChanMap.clear();
  if( FillDetMap(detmap, THaDetMap::kFillRefIndex, here) <= 0 ) {
    return kInitError;  // Error already printed by FillDetMap
  }

  // Step through the modules already filled and find out how many reference
  // channels we have (and also, make these modules TDCs)
  nref_chans = 0;
  for( UShort_t imod = 0; imod < fDetMap->GetSize(); imod++ ) {
    THaDetMap::Module *d = fDetMap->GetModule( imod );
    d->MakeTDC();
    if(d->refindex == -1) {
      nref_chans += (d->hi - d->lo)+1;
    }
  }

  // Now that we got the number of reference channels, let's make sure
  // the user specified the resolution of those channels
  if(ref_ch_res.size() == 1) {
    for(int i = 0; i < nref_chans; i++) {
      new((*fRefCh)[i]) SBSScintPMT(1.0,0,ref_ch_res[0]);
    }
  } else if (ref_ch_res.size() == nref_chans) {
    for(int i = 0; i < nref_chans; i++) {
      new((*fRefCh)[i]) SBSScintPMT(1.0,0,ref_ch_res[i]);
    }
  } else {
    Error( Here(here), "Malformed ref_ch_res in database. %d entries provided"
        " but expected either 1 or %d",ref_ch_res.size(),nref_chans);
    return kInitError;

  }

  fNelem = fDetMap->GetTotNumChan();
  if(fNelem != npmt + nref_chans) {
    Error( Here(here), "Number of crate module channels (%d) "
        "inconsistent with number of PMTs(2*nbars=%d) + refeference channels (%d)",
        fNelem, 2*nbars, nref_chans );
    return kInitError;
  }

  // Now fill the ADC detmap if specified
  if(detmap_adc.size() > 0 ) {
    if(FillDetMap(detmap_adc, THaDetMap::kDoNotClear, here) <= 0 ) {
      return kInitError;  // Error already printed by FillDetMap
    } else {
      fNelem = fDetMap->GetTotNumChan();
      Int_t nadc = fNelem - npmt - nref_chans;
      if(nadc != npmt) {
        Error( Here(here), "Number of ADC crate module channels (%d) "
            "inconsistent with number of PMTs(2*nbars=%d)",
            nadc, 2*nbars);
        return kInitError;
      }
      for( UShort_t imod = 0; imod < fDetMap->GetSize(); imod++ ) {
        THaDetMap::Module *d = fDetMap->GetModule( imod );
        d->MakeADC();
      }
    }
  }

  typedef std::pair<const char*,int> t_v;
  std::vector<t_v> vars;
  vars.push_back(t_v("bar_geom",6));
  vars.push_back(t_v("speed_of_light",1));
  vars.push_back(t_v("attenuation",1));
  for(int i = 0; i < 2; i++) {
    const char *name = (i==0?"left":"right");
    vars.push_back(t_v(Form("%s_pedestal",name),1));
    vars.push_back(t_v(Form("%s_gain",name),1));
    vars.push_back(t_v(Form("%s_calib",name),4));
    vars.push_back(t_v(Form("%s_toff",name),1));
    vars.push_back(t_v(Form("%s_walkcor",name),1));
    vars.push_back(t_v(Form("%s_walkexp",name),1));
  }

  std::vector<DBRequest> calib_request;
  int nvars = vars.size();
  std::vector<Float_t> db_read[nvars];
  int nvals = 0;
  for(int i = 0; i < nvars; i++) {
    calib_request.push_back({vars[i].first,&db_read[i],kFloatV});
    nvals += vars[i].second;
  }
  calib_request.push_back({0});
  
  // Make the databse request
  err = LoadDB( file, date, calib_request.data(), fPrefix );
  if(err) return err;

  // Parse through each database variable and fill the bar properties
  int first,last;
  Float_t db[nbars][nvals];
  int idx = 0;
  for(int i = 0; i < nvars; i++) {
    int ni = vars[i].second; // Number of elements per bar/pmt
    int n = db_read[i].size(); // Vector size
    if(n == ni) {
      for(Int_t ibar = 0; ibar < nbars; ibar++ ) {
        for(Int_t j = 0; j < ni; j++) {
          db[ibar][idx+j] = db_read[i][j];
        }
      }
      idx+=ni;
    } else if ( n == nbars*ni ) {
      for(Int_t ibar = 0; ibar < nbars; ibar++) {
        for(int j = 0; j < ni; j++) {
          db[ibar][idx+j] = db_read[i][ibar*ni+j];
        }
      }
      idx+=ni;
    } else {
      Error(Here(here), "Malformed database for %s: Has '%d' entries, and "
         " expected either %d*nbars=%d or %d to apply to all bars.",
          vars[i].first,n,ni,ni*nbars,ni);
      return kInitError;
    }
  }

  // A quick sanity check for geometry. If bar_geom only has 6 values,
  // then the origins of all the bars were set the same. Fix that now
  if(db_read[0].size() == 6) {
    float x[3] = { db[0][0], db[0][1], db[0][2] };
    for(int ibar = 1; ibar <nbars; ibar++) {
      for(int j = 0; j < 3; j++) {
        x[j] += dxyz[j];
        db[ibar][j] = x[j];
      }
    };
  }

  // Next, after reading the entire DB, let's create the instances of
  // those ScintBars
  fRefCh->Clear();
  fBars->Clear();
  for(int i = 0; i < nbars; i++) {
    new((*fBars)[i]) SBSScintBar(
        db[i][0],db[i][1], db[i][2], db[i][3], db[i][4], db[i][5],
        db[i][6],db[i][7],
        db[i][8],db[i][9], db[i][10], // skip 11, 12, 13  for now
        db[i][14],db[i][15], // Skip 16 it does not go in constructor

        db[i][17],db[i][18],db[i][19], // Skip 20, 21, 22 for now
        db[i][23],db[i][24], // Skip 25, it does not go in the constructor
        i,
        db[i][11],db[i][12],db[i][13],
        db[i][20],db[i][21],db[i][22]
        );
    (GetBar(i)->GetLPMT())->SetTimeWExp(db[i][16]);
    (GetBar(i)->GetRPMT())->SetTimeWExp(db[i][25]);
  }

  fNBars= GetNBars();
  if (fLE) {
    Warning(Here(here),"Re-initializing: detectors must be the SAME SIZE!");
  } else {
    fLE    = new Double_t[fNBars];
    fRE    = new Double_t[fNBars];
    fLT    = new Double_t[fNBars];
    fRT    = new Double_t[fNBars];
    fLrawA = new Double_t[fNBars];
    fRrawA = new Double_t[fNBars];
    fLpedcA= new Double_t[fNBars];
    fRpedcA= new Double_t[fNBars];

    fLTcounter= new Int_t[fNBars];
    fRTcounter= new Int_t[fNBars];
    hitcounter= new Int_t[fNBars];
    Energy = new Double_t[fNBars];
    TDIFF  = new Double_t[fNBars];
    T_tot  = new Double_t[fNBars];
    TOF    = new Double_t[fNBars];
    Yt_pos = new Double_t[fNBars];
    Ya_pos = new Double_t[fNBars];
    Y_pred = new Double_t[fNBars];
    Y_dev  = new Double_t[fNBars];
    fAngle = new Double_t[fNBars];
    fLtIndex = new Int_t[fNBars];
    fRtIndex = new Int_t[fNBars];
    fLaIndex = new Int_t[fNBars];
    fRaIndex = new Int_t[fNBars];
  }

  return (!err?err:kOK);
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::LoadDBPMT(FILE* file, const TDatime &date, bool is_left)
{
  std::vector<Float_t> ped,gain,toff, walkcor, walkexp;
  const char *prefix = Form("%s%s_",fPrefix,is_left?"left":"right");

  DBRequest pmt_request[] = {
    { "pedestal", &ped,     kFloatV },
    { "gain",     &gain,    kFloatV },
    { "toff",     &toff,    kFloatV },
    { "walkcor",  &walkcor, kFloatV },
    { "walkexp",  &walkexp, kFloatV },
    { 0 } ///< Request must end in a NULL
  };
  Int_t err = LoadDB( file, date, pmt_request, prefix );
  if(err) return err;

  // Now loop through the read values and set properties for the given bars;
  int first,last;
  std::vector<Float_t>::iterator it;
  it = ped.begin();
  if(ped.size() % 3 == 0) {
    while(it != ped.end() ) {
      first = (*it++);
      last  = (*it++);
      if(last == -1) {
        last = GetNBars();
      }
      for(int i = first; i <= last; i++) {

      }
    }
  }

  return -1;

}


//_____________________________________________________________________________

Int_t SBSTimingHodoscope::ReadDatabaseOld( const TDatime& date )
{
	// Read this detector's parameters from the database file 'fi'.
	// This function is called by THaNonTrackingDetectorBase::Init() once at the
	// beginning of the analysis.
	// 'date' contains the date/time of the run being analyzed.


	FILE* file = OpenFile( date );

	if( !file ) return kFileError;

	// Use default values until ready to read from database

	static const char* const here = "ReadDatabase";
	const int LEN = 200;
	char buff[LEN];
	char cbuff[LEN];	//commment buffer
	Int_t i;

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tStart reading Database");
#endif//#if DEBUG_LEVEL>=3

	// Build the search tag and find it in the file. Search tags
	// are of form [ <prefix> ], e.g. [ N.bar.n1 ].
	TString line;

	TString prefix=fPrefix;
	prefix.Chop();  // remove trailing dot of prefix
	TString tag = Form("[%s.%s]",prefix.Data(),  "cratemap.tdc"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	bool found = false;
	while (!found && fgets (buff, LEN, file) != NULL) {
		char* buf = ::Compress(buff);  //strip blanks
		line = buf;
		delete [] buf;
#if DEBUG_LEVEL>=5//start show info
		Info(Here(here),"\tTrying to search line <%s>", line.Data());
#endif
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
#if DEBUG_LEVEL>=5//start show info
		Info(Here(here),"\tafter line.Chop() <%s>", line.Data());
#endif
		line.ToLower();
		if ( tag == line ) 
			found = true;
	}
	if( !found ) {
		Error(Here(here), "Database entry \"%s\" not found!", tag.Data() );
		fclose(file);
		return kInitError;
	}

	//Found the entry for this plane
	//first are definitions of the reference channels
	//then the definitions of the tdc channels for individiual PMTs
	//
	Int_t nLPMTs = 0;    // Number of PMTs to create
	Int_t nRPMTs = 0;    // Number of PMTs to create

	Int_t nRefCh = 0;    // Number of ref.channels to create
	Int_t prev_first = 0, prev_nPMTs = 0;
	// Set up the detector map
	fDetMap->Clear();

	Int_t crate, slot, lo, hi, model, refindex;
	Int_t lolo;
	crate = 0 ;
	// Get crate, slot, low channel and high channel from file
	// for the reference channels
	while (crate!=-1) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d%d%d%d%d", &crate, &slot, &lo, &hi, &model, &refindex ) != 6 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 9 reading %s : %s", tag.Data(),buff );
			fclose(file);
			return kInitError;
		}
		if (crate!=-1) {
			int first = prev_first + prev_nPMTs;
			// Add module to the detector map
			if ( fDetMap->AddModule(crate, slot, lo, hi, first, model, refindex) < 0 ) {
				Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
					THaDetMap::kDetMapSize);
				fclose(file);
				return kInitError;
			}
			prev_first = first;
			prev_nPMTs = (hi - lo + 1 );
			nRefCh += prev_nPMTs;
		}
	}

	// Get crate, slot, low channel and high channel from file
	// for the left PMTs
	crate = 0;
	while (crate!=-1) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d%d%d%d%d", &crate, &slot, &lo, &hi, &model, &refindex ) != 6 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 10 reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
		if (crate!=-1) {
			if (lo<=hi) {
				int first = prev_first + prev_nPMTs;
				// Add module to the detector map
				if ( fDetMap->AddModule(crate, slot, lo, hi, first, model, refindex) < 0 ) {
					Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
						THaDetMap::kDetMapSize);
					fclose(file);
					return kInitError;
				}
				prev_first = first;
				prev_nPMTs = (hi - lo + 1 );
				nLPMTs += prev_nPMTs;
			} else {
				lolo = lo ;
				while (lolo>=hi) {
					int first = prev_first + prev_nPMTs;
					if ( fDetMap->AddModule(crate, slot, lolo, lolo, first, model, refindex) < 0 ) {
						Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
							THaDetMap::kDetMapSize);
						fclose(file);
						return kInitError;
					}
					prev_first = first;
					prev_nPMTs =  1 ;
					nLPMTs += prev_nPMTs;
					lolo -= 1 ;
				}
			}
		}
	}

	// Get crate, slot, low channel and high channel from file
	// for the right PMTs
	crate = 0;
	while (crate!=-1) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d%d%d%d%d", &crate, &slot, &lo, &hi, &model, &refindex ) != 6 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 11 reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
		if (crate!=-1) {
			if (lo<=hi) {
				int first = prev_first + prev_nPMTs;
				// Add module to the detector map
				if ( fDetMap->AddModule(crate, slot, lo, hi, first, model, refindex) < 0 ) {
					Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
						THaDetMap::kDetMapSize);
					fclose(file);
					return kInitError;
				}
				prev_first = first;
				prev_nPMTs = (hi - lo + 1 );
				nRPMTs += prev_nPMTs;
			} else {
				lolo = lo ;
				while (lolo>=hi) {
					int first = prev_first + prev_nPMTs;
					if ( fDetMap->AddModule(crate, slot, lolo, lolo, first, model, refindex) < 0 ) {
						Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
							THaDetMap::kDetMapSize);
						fclose(file);
						return kInitError;
					}
					prev_first = first;
					prev_nPMTs =  1 ;
					nRPMTs += prev_nPMTs;
					lolo -= 1 ;
				}
			}
		}
	}

	//Warning Thresholds
	tag = Form("[%s.%s]",prefix.Data(),  "WarningThreshold"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	found=false;
	rewind(file);
	do {
		if (fgets (buff, LEN, file) == NULL ) {
			Error(Here(here), "Database entry \"%s\" not found!", tag.Data() );
			fclose(file);
			return kInitError;
		}
		char* buf = ::Compress(buff);  //strip blanks
		line = buf;
		delete [] buf;
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
		line.ToLower();
		if ( tag == line ) 
			found = true;
	} while (!found);

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

	fgets( buff, LEN, file );
	if( sscanf( buff, "%lg", &fErrorReferenceChRateWarningThreshold ) != 1 ) {
		if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
		Error( Here(here), "Error 12 reading %s : %s", tag.Data(),buff );
		fclose(file);
		return kInitError;
	}

	DEBUG_INFO(Here(here),
		"fErrorReferenceChRateWarningThreshold=%f"
		,fErrorReferenceChRateWarningThreshold);

	if (fErrorReferenceChRateWarningThreshold<0 
		or fErrorReferenceChRateWarningThreshold>1)
	{
		DEBUG_WARNING(Here(here)
			,"Invalid value (=%f) from database section %s for Threshold of error reference"
			,fErrorReferenceChRateWarningThreshold,tag.Data());
	}


	// now we search for the ADC detector map
	tag = Form("[%s.%s]",prefix.Data(),  "cratemap.adc"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	found = false;
	rewind(file);
	while (!found && fgets (buff, LEN, file) != NULL) {
		char* buf = ::Compress(buff);  //strip blanks
		line = buf;
		delete [] buf;
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
		line.ToLower();
		if ( tag == line ) 
			found = true;
	}
	if( !found ) {
		Error(Here(here), "Database entry \"%s\" not found!", tag.Data() );
		fclose(file);
		return kInitError;
	}


	//Found the entry for this plane
	//first the definition for the left adc channels then for the right side
	//
	Int_t nLPMTadc = 0;    // Number of PMTs to create
	Int_t nRPMTadc = 0;    // Number of PMTs to create

	crate = 0;
	while (crate!=-1) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d%d%d%d", &crate, &slot, &lo, &hi, &model ) != 5 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 12 reading %s : %s", tag.Data(),buff );
			fclose(file);
			return kInitError;
		}
		if (crate!=-1) {
			if (lo<=hi) {
				int first = prev_first + prev_nPMTs;
				// Add module to the detector map
				if ( fDetMap->AddModule(crate, slot, lo, hi, first, model ) < 0 ) {
					Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
						THaDetMap::kDetMapSize);
					fclose(file);
					return kInitError;
				}
				prev_first = first;
				prev_nPMTs = (hi - lo + 1 );
				nLPMTadc += prev_nPMTs;
			} else {
				lolo = lo ;
				while (lolo>=hi) {
					int first = prev_first + prev_nPMTs;
					if ( fDetMap->AddModule(crate, slot, lolo, lolo, first, model) < 0 ) {
						Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
							THaDetMap::kDetMapSize);
						fclose(file);
						return kInitError;
					}
					prev_first = first;
					prev_nPMTs =  1 ;
					nLPMTadc += prev_nPMTs;
					lolo -= 1 ;
				}
			}
		}
	}

	// Get crate, slot, low channel and high channel from file
	// for the right PMTs
	crate = 0;
	while (crate!=-1) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d%d%d%d", &crate, &slot, &lo, &hi, &model ) != 5 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 13 reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
		if (crate!=-1) {
			if (lo<=hi) {
				int first = prev_first + prev_nPMTs;
				// Add module to the detector map
				if ( fDetMap->AddModule(crate, slot, lo, hi, first, model ) < 0 ) {
					Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
						THaDetMap::kDetMapSize);
					fclose(file);
					return kInitError;
				}
				prev_first = first;
				prev_nPMTs = (hi - lo + 1 );
				nRPMTadc += prev_nPMTs;
			} else {
				lolo = lo ;
				while (lolo>=hi) {
					int first = prev_first + prev_nPMTs;
					if ( fDetMap->AddModule(crate, slot, lolo, lolo, first, model ) < 0 ) {
						Error( Here(here), "Too many DetMap modules (maximum allowed - %d).", 
							THaDetMap::kDetMapSize);
						fclose(file);
						return kInitError;
					}
					prev_first = first;
					prev_nPMTs =  1 ;
					nRPMTadc += prev_nPMTs;
					lolo -= 1 ;
				}
			}
		}
	}

	if ( (nRPMTadc != nLPMTadc) || (nRPMTs != nLPMTs) || 
		(nLPMTadc != nLPMTs) || (nRPMTadc != nRPMTs) ) 
	{
		Error( Here(here), 
			" Database corrupted, mismatch in number of ADC or TDC channels.");
		fclose(file);
		return kInitError;
	}

	tag = Form("[%s.%s]",prefix.Data(),  "calib"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	found=false;
	rewind(file);
	do {
		if (fgets (buff, LEN, file) == NULL ) {
			Error(Here(here), "Database entry \"%s\" not found!", tag.Data() );
			fclose(file);
			return kInitError;
		}
		char* buf = ::Compress(buff);  //strip blanks
		line = buf;
		delete [] buf;
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
		line.ToLower();
		if ( tag == line ) 
			found = true;
	} while (!found);



	Int_t prevfirst=0;
	Int_t prevlast=-1;
	int first=0;
	int last=-1;
	Double_t lres=1;
	Double_t rres=1;

	fRefCh->Clear();
	fBars->Clear();

	while (( last<nRefCh-1 )&&( first<nRefCh) ) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d",&first,&last) != 2 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 14 reading %s : %s", tag.Data(),buff );
			fclose(file);
			return kInitError;
		}
		if (first>prevlast+1) { 
			for (i=prevlast+1;i<first;i++) {
				new((*fRefCh)[i]) SBSScintPMT(1.,0,lres);
#if DEBUG_LEVEL>=4//massive info
				Info(Here(here),"\tlres:new((*fRefCh)[%d]) SBSScintPMT(1.0,0,%lg)",i, lres);
#endif//#if DEBUG_LEVEL>=4
			}
		}
		if (last<first) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 15 reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		} else {
			if (first<nRefCh) {
				if (last>=nRefCh) { last=nRefCh-1; }

				//jump throug comment lines
				while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

				fgets( buff, LEN, file );
				if( sscanf( buff, "%lg",&rres) != 1 ) {
					if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
					Error( Here(here), "Error 16 reading %s : %s", tag.Data(), buff );
					fclose(file);
					return kInitError;
				}
				for (i=first;i<=last;i++) {
					new((*fRefCh)[i]) SBSScintPMT(1.0,0,rres);
#if DEBUG_LEVEL>=4//massive info
					Info(Here(here),"\trres:new((*fRefCh)[%d]) SBSScintPMT(1.0,0,%lg)",i, rres);
#endif//#if DEBUG_LEVEL>=4
				}
			}
			prevfirst=first;
			prevlast=last;
		}
	}

	//construct parts
	Double_t x=0,y=0,z=0,dx=0,dy=0,dz=0,xw=0,yw=0,zw=0,c=3e8,att=0;
	Double_t lgain=1,ltoff=0,lwalk=0;
	Int_t lped=0;
	Double_t rgain=1,rtoff=0,rwalk=0;
	Int_t rped=0;
	rres=1;
	lres=1;
	Double_t lwrapa=0.;
	Double_t rwrapa=0.;
	Int_t llowtdclim=0, luptdclim=65536;
	Int_t rlowtdclim=0, ruptdclim=65536;
	prevfirst=0;
	prevlast=-1;
	first=0;
	last=-1;

	while (( last<nLPMTs-1 )&&( first<nLPMTs) ) {

		//jump throug comment lines
		while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

		fgets( buff, LEN, file );
		if( sscanf( buff, "%d%d",&first,&last) != 2 ) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 17 reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
		if (first>prevlast+1) { 
			for (i=prevlast+1;i<first;i++) {
#if DEBUG_LEVEL>=4//massive info
				Info(Here(here),"\tnew((*fBars)[%d]) SBSScintBar",i);
#endif//#if DEBUG_LEVEL>=4
				new((*fBars)[i]) SBSScintBar(x,y,z,xw,yw,zw,c,att,
					lgain,lped,lres,ltoff,lwalk,
					rgain,rped,rres,rtoff,rwalk,
					i,
					llowtdclim,luptdclim,lwrapa,
					rlowtdclim,ruptdclim,rwrapa);
				x=x+dx; y=y+dy; z=z+dz;
			}
		}
		if (last<first) {
			if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
			Error( Here(here), "Error 18 reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		} else {
			if (first<nLPMTs) {

				if (last>=nLPMTs) { last=nLPMTs-1; }

				//jump throug comment lines
				while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

				fgets( buff, LEN, file );
				if( sscanf( buff, "%lg%lg%lg%lg%lg%lg",&x,&y,&z,&dx,&dy,&dz) != 6 ) {
					if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
					Error( Here(here), "Error 1 reading %s : %s", tag.Data(), buff );
					fclose(file);
					return kInitError;
				}

				//jump throug comment lines
				while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

				fgets( buff, LEN, file );
				if( sscanf( buff, "%lg%lg%lg",&xw,&yw,&zw) != 3 ) {
					if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
					Error( Here(here), "Error 2 reading %s : %s", tag.Data(), buff );
					fclose(file);
					return kInitError;
				}

				//jump throug comment lines
				while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

				fgets( buff, LEN, file );
				if( sscanf( buff, "%lg%lg",&c,&att) != 2 ) {
					if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
					Error( Here(here), "Error 3 reading %s : %s", tag.Data(), buff );
					fclose(file);
					return kInitError;
				}
#if DEBUG_LEVEL>=4//massive info
				Info(Here(here),"\tc=%lg,att=%lg",c,att);
#endif//#if DEBUG_LEVEL>=4

				//jump throug comment lines
				while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

				fgets( buff, LEN, file );
				if( sscanf( buff, "%lg%d%lg%lg%lg%d%d%lg",&lgain,&lped,&lres,&ltoff,&lwalk,&llowtdclim,&luptdclim,&lwrapa) != 8 ) {
					if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
					Error( Here(here), "Error 4 reading %s : %s", tag.Data(), buff );
					fclose(file);
					return kInitError;
				}

				//jump throug comment lines
				while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

				fgets( buff, LEN, file );
				if( sscanf( buff, "%lg%d%lg%lg%lg%d%d%lg",&rgain,&rped,&rres,&rtoff,&rwalk,&rlowtdclim,&ruptdclim,&rwrapa) != 8 ) {
					if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
					Error( Here(here), "Error 5 reading %s : %s", tag.Data(), buff );
					fclose(file);
					return kInitError;
				}

				for (i=first;i<=last;i++) {
					new((*fBars)[i]) SBSScintBar(x,y,z,xw,yw,zw,c,att,
						lgain,lped,lres,ltoff,lwalk,
						rgain,rped,rres,rtoff,rwalk,
						i,
						llowtdclim,luptdclim,lwrapa,
						rlowtdclim,ruptdclim,rwrapa
						);
					x=x+dx; y=y+dy; z=z+dz;
				}
			}
			prevfirst=first;
			prevlast=last;      
		}
	}
	//// optional parts in the database:
	//// Threshold a standalone number at the end of the calib section
	////jump throug comment lines
	//while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

	//fgets( buff, LEN, file );
	//Double_t v;
	//if( sscanf( buff, "%lg",&v ) == 1 ) {
	//	fThreshold = v;
	//}

	// now for the tables: rewind the file and start again

	// to read in tables of pedestals

	Double_t* values = new Double_t[GetNBars()+10];
	Int_t j;
	tag = Form("[%s.%s]",prefix.Data(),  "left_pedestals"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nLPMTs)&&(last>=first)&&(last<nLPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				lped = (Int_t) values[j];
				(GetBar(i)->GetLPMT())->SetPed(lped);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "right_pedestals"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nRPMTs)&&(last>=first)&&(last<nRPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				rped = (Int_t) values[j];
				(GetBar(i)->GetRPMT())->SetPed(rped);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "left_gain"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nLPMTs)&&(last>=first)&&(last<nLPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetLPMT())->SetGain(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "right_gain"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nRPMTs)&&(last>=first)&&(last<nRPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetRPMT())->SetGain(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "left_toff"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nLPMTs)&&(last>=first)&&(last<nLPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetLPMT())->SetTOffset(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "right_toff"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nRPMTs)&&(last>=first)&&(last<nRPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetRPMT())->SetTOffset(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "speed_of_light"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<GetNBars())&&(last>=first)&&(last<GetNBars())) {
			j=0;
			for (i=first; i<=last; i++) {
				GetBar(i)->SetC(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "attenuation"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<GetNBars())&&(last>=first)&&(last<GetNBars())) {
			j=0;
			for (i=first; i<=last; i++) {
				GetBar(i)->SetAtt(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "left_walkcor"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nLPMTs)&&(last>=first)&&(last<nLPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetLPMT())->SetTimeWalk(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "right_walkcor"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nRPMTs)&&(last>=first)&&(last<nRPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetRPMT())->SetTimeWalk(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "left_walkexp"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nLPMTs)&&(last>=first)&&(last<nLPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetLPMT())->SetTimeWExp(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	tag = Form("[%s.%s]",prefix.Data(),  "right_walkexp"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();

	rewind(file);
	while (GetTable(file,tag,values,GetNBars(),&first,&last)==0) {
		if ((first>=0)&&(first<nRPMTs)&&(last>=first)&&(last<nRPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				(GetBar(i)->GetRPMT())->SetTimeWExp(values[j]);
				j++;
			}      
		} else {
			Error( Here(here), "Error reading %s : %s", tag.Data(), buff );
			fclose(file);
			return kInitError;
		}
	}

	delete [] values;

	// now for the geometry
	tag = Form("[%s.%s]",prefix.Data(),  "bar_geom"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	Int_t maxvs = 6*GetNBars();
	values = new Double_t[maxvs];  // x y z dx dy dz

	rewind(file);
	while (GetTable(file,tag,values,maxvs,&first,&last)==0) {
		// put last back into bar number
		if ((first>=0)&&(first<nRPMTs)&&(last>=first)&&(last<nRPMTs)) {
			j=0;
			for (i=first; i<=last; i++) {
				SBSScintBar* b = GetBar(i);
				if (!b) {
					Error( Here(here), "Error setting geometry, found at entry %d for bar %d", j,i );
					fclose(file);
					return kInitError;
				}
				b->SetXPos(values[j++]);
				b->SetYPos(values[j++]);
				b->SetZPos(values[j++]);
				b->SetXWidth(values[j++]);
				b->SetYWidth(values[j++]);
				b->SetZWidth(values[j++]);
			}      
		} else {
			Error( Here(here), "Error reading %s : %s, first=%d, last=%d, nRPMTs=%d", tag.Data(), buff,first,last,nRPMTs );
			fclose(file);
			return kInitError;
		}
	}

	delete [] values;

	fNBars= GetNBars();
	if (fLE) {
		Warning(Here(here),"Re-initializing: detectors must be the SAME SIZE!");
	} else {
		fLE    = new Double_t[fNBars];
		fRE    = new Double_t[fNBars];
		fLT    = new Double_t[fNBars];
		fRT    = new Double_t[fNBars];
		fLrawA = new Double_t[fNBars];
		fRrawA = new Double_t[fNBars];
		fLpedcA= new Double_t[fNBars];
		fRpedcA= new Double_t[fNBars];

		fLTcounter= new Int_t[fNBars];
		fRTcounter= new Int_t[fNBars];
		hitcounter= new Int_t[fNBars];
		Energy = new Double_t[fNBars];
		TDIFF  = new Double_t[fNBars];
		T_tot  = new Double_t[fNBars];
		TOF    = new Double_t[fNBars];
		Yt_pos = new Double_t[fNBars];
		Ya_pos = new Double_t[fNBars];
		Y_pred = new Double_t[fNBars];
		Y_dev  = new Double_t[fNBars];
		fAngle = new Double_t[fNBars];
		fLtIndex = new Int_t[fNBars];
		fRtIndex = new Int_t[fNBars];
		fLaIndex = new Int_t[fNBars];
		fRaIndex = new Int_t[fNBars];
	}


	///////////////////////// geometry ///////////////////////////
	tag = Form("[%s.%s]",prefix.Data(),  "geometry"  );

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	found = false;

	rewind(file);
	while (!found && fgets (buff, LEN, file) != NULL) {
		char* buf = ::Compress(buff);  //strip blanks
		line = buf;
		delete [] buf;
#if DEBUG_LEVEL>=5//start show info
		Info(Here(here),"\tTrying to search line <%s>", line.Data());
#endif
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
#if DEBUG_LEVEL>=5//start show info
		Info(Here(here),"\tafter line.Chop() <%s>", line.Data());
#endif
		line.ToLower();
		if ( tag == line ) 
			found = true;
	}
	if( !found ) {
		Error(Here(here), "Database entry \"%s\" not found!", tag.Data() );
		fclose(file);
		return kInitError;
	}

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));

	Double_t xtmp,ytmp,ztmp;
	fgets( buff, LEN, file );
	if( sscanf( buff, "%lg%lg%lg",&xtmp,&ytmp,&ztmp) != 3 ) {
		if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
		Error( Here(here), "Error 4 reading %s (fOrigin): %s", tag.Data(), buff );
		fclose(file);
		return kInitError;
	}
	fOrigin.SetXYZ(xtmp,ytmp,ztmp);

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));

	fgets( buff, LEN, file );
	if( sscanf( buff, "%lg%lg%lg",&xtmp,&ytmp,&ztmp) != 3 ) {
		if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
		Error( Here(here), "Error 4 reading %s (fXax): %s", tag.Data(), buff );
		fclose(file);
		return kInitError;
	}
	fXax.SetXYZ( xtmp,ytmp,ztmp );

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));

	fgets( buff, LEN, file );
	if( sscanf( buff, "%lg%lg%lg",&xtmp,&ytmp,&ztmp) != 3 ) {
		if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
		Error( Here(here), "Error 4 reading %s (fZax): %s", tag.Data(), buff );
		fclose(file);
		return kInitError;
	}
	fYax.SetXYZ( xtmp,ytmp,ztmp );
	fZax = fXax.Cross(fYax);


	///////////////////////// hit_acceptance ///////////////////////////

	tag = Form("[%s.%s]",prefix.Data(),  "hit_acceptance"  );

	rewind(file);

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"\tTrying to read in database section %s", tag.Data());
#endif

	tag.ToLower();
	found = false;
	while (!found && fgets (buff, LEN, file) != NULL) {
		char* buf = ::Compress(buff);  //strip blanks
		line = buf;
		delete [] buf;
#if DEBUG_LEVEL>=5//start show info
		Info(Here(here),"\tTrying to search line <%s>", line.Data());
#endif
		if( line.EndsWith("\n") ) line.Chop();  //delete trailing newline
#if DEBUG_LEVEL>=5//start show info
		Info(Here(here),"\tafter line.Chop() <%s>", line.Data());
#endif
		line.ToLower();
		if ( tag == line ) 
			found = true;
	}
	if( !found ) {
		Error(Here(here), "Database entry \"%s\" not found!", tag.Data() );
		fclose(file);
		return kInitError;
	}

	//jump throug comment lines
	while ( ReadNumberSignStartComment( file, cbuff, LEN ));	

	//#accepatble time difference between 2 hits in E and dE plane is smaller
	//#time difference=TOF(dE)-TOF(E)
	fgets( buff, LEN, file );
	if( sscanf( buff, "%lg%lg",&fTrackAcceptanceDx,&fTrackAcceptanceDy) != 2 ) {
		if( *buff ) buff[strlen(buff)-1] = 0; //delete trailing newline
		Error( Here(here), "Error 4 reading %s : %s", tag.Data(), buff );
		fclose(file);
		return kInitError;
	}

#if DEBUG_LEVEL>=3//start show info
	Info(Here(here),"Printing Origin and x,y,z axis vector:");
	fOrigin.Print();
	fXax.Print();
	fYax.Print();
	fZax.Print();

	TString sDebugOutput;
	sDebugOutput=GetName();
	sDebugOutput+=" Database read in successfully with:";
	sDebugOutput+="\n \tTrackAcceptanceDx\t= ";	sDebugOutput+=fTrackAcceptanceDx;
	sDebugOutput+="\n \tTrackAcceptanceDy\t= ";	sDebugOutput+=fTrackAcceptanceDy;
	sDebugOutput+="\n";
	Info(Here(here),sDebugOutput.Data());

	Info(Here(here),"Printing Detector Map:");
	fDetMap->Print();

#endif	



	fclose(file);
	return kOK;
}


//_____________________________________________________________________________
Int_t SBSTimingHodoscope::DefineVariables( EMode mode )
{
	// Initialize global variables and lookup table for decoder

	if( mode == kDefine && fIsSetup ) return kOK;
	fIsSetup = ( mode == kDefine );

	// Register variables in global list

	RVarDef vars[] = {
		{ "refchokay",         "All relevant ref channels are okay",        "AreRefChOkay()" },
		//      { "nbar",              "Number of bars",                            "GetNBars()"},  
		//      { "bar_num",           "bar number of bar",                         "fBars.SBSScintBar.GetBarNum()"},
		//      { "bar_xpos",          "x position of bar",                         "fBars.SBSScintBar.GetXPos()"},
		//      { "bar_ypos",          "y position of bar",                         "fBars.SBSScintBar.GetYPos()"},
		//      { "bar_zpos",          "z position of bar",                         "fBars.SBSScintBar.GetZPos()"},
		//      { "bar_xwidth",        "width_x of bar",                            "fBars.SBSScintBar.GetXWidth()"},
		//      { "bar_ywidth",        "width_y of bar",                            "fBars.SBSScintBar.GetYWidth()"},
		//      { "bar_zwidth",        "width_z of bar",                            "fBars.SBSScintBar.GetZWidth()"},
		//      { "bar_c",             "speed of light in bar",                     "fBars.SBSScintBar.GetC()"},
		//      { "bar_att",           "Attenuation length of bar",                 "fBars.SBSScintBar.GetAtt()"},
		//      { "bar_type",          "Type of bar",                               "fBars.SBSScintBar.GetBarType()"},
		//      { "bar_nd",            "Type of bar",                               "fBars.SBSScintBar.GetBarNum_nd()"},
		// raw-data information
		{ "nrefhit",           "Number of ref hits",                        "GetNRefHits()"},
		{ "ref_bar",           "bars in fRefHits",                          "fRefHits.SBSTdcHit.GetBarNum()"},
		{ "ref_tdc",           "raw time in fRefHits",                      "fRefHits.SBSTdcHit.GetRawTime()"},
		{ "ref_time",          "time in fRefHits",                          "fRefHits.SBSTdcHit.GetTime()"},
		{ "nlthit",            "Number of lt hits",                         "GetNLtHits()"},
		{ "lthit_bar",          "bars in fLtHits",                           "fLtHits.SBSTdcHit.GetBarNum()"},
		{ "lthit_tdc",          "raw time in fLtHits",                       "fLtHits.SBSTdcHit.GetRawTime()"},
		{ "lthit_time",         "time in fLtHits",                           "fLtHits.SBSTdcHit.GetTime()"},
		{ "lthit_side",         "side in fLtHits",                           "fLtHits.SBSTdcHit.GetSide()"},
		{ "nrthit",            "Number of rt hits",                         "GetNRtHits()"},
		{ "rthit_bar",          "bars in fRtHits",                           "fRtHits.SBSTdcHit.GetBarNum()"},
		{ "rthit_tdc",          "raw time in fRtHits",                       "fRtHits.SBSTdcHit.GetRawTime()"},
		{ "rthit_time",         "time in fRtHits",                           "fRtHits.SBSTdcHit.GetTime()"},
		{ "rthit_side",         "side in fRtHits",                           "fRtHits.SBSTdcHit.GetSide()"},
		{ "nlahit",            "Number of la hits",                         "GetNLaHits()"},
		{ "lahit_bar",         "bars in fLaHits",                           "fLaHits.SBSAdcHit.GetBarNum()"},
		{ "lahit_adc",         "raw amplitude in fLaHits",                  "fLaHits.SBSAdcHit.GetRawAmpl()"},
		{ "lahit_ap",          "pedistal corrected ampl in fLaHits",        "fLaHits.SBSAdcHit.GetAmplPedCor()"},
		{ "lahit_ac",          "Amplitude in fLaHits",                      "fLaHits.SBSAdcHit.GetAmpl()"}, 
		{ "lahit_side",        "side of bar in fLaHits",                    "fLaHits.SBSAdcHit.GetSide()"}, 
		{ "nrahit",            "Number of Rahits",                          "GetNRaHits()"},
		{ "rahit_bar",         "bars in fRaHits",                           "fRaHits.SBSAdcHit.GetBarNum()"},
		{ "rahit_adc",         "raw amplitude in fRaHits",                  "fRaHits.SBSAdcHit.GetRawAmpl()"},
		{ "rahit_ap",          "pedistal corrected ampl in fRaHits",        "fRaHits.SBSAdcHit.GetAmplPedCor()"},
		{ "rahit_ac",          "Amplitude in fRaHits",                      "fRaHits.SBSAdcHit.GetAmpl()"}, 
		{ "rahit_side",        "side of bar in fRaHits",                    "fRaHits.SBSAdcHit.GetSide()"},
		// complete reconstructed hits
		{ "nhit",              "Number of hits",                            "GetNHits()"},
		{ "hit_bar",           "bars in fHits",                             "fHits.SBSScintHit.GetBarNum()"},
		{ "hit_xpos",          "X position in fHits",                       "fHits.SBSScintHit.GetHitXPos()"},
		{ "hit_ypos",          "Y position in fHits",                       "fHits.SBSScintHit.GetHitYPos()"},
		{ "hit_tof",           "TOF in fHits = .5*(Lt + Rt)",               "fHits.SBSScintHit.GetHitTOF()"},
		{ "hit_Edep",          "Edep in fHits",                             "fHits.SBSScintHit.GetHitEdep()"},
		{ "hit_tdiff",         "Time diff in fHits=.5*(Rt - Lt)",           "fHits.SBSScintHit.GetHitTdiff()"},    
#if BUILD_PARTIAL_HIT 
		// partial hits -- probably will NOT use
		{ "nparthit",          "Number of partial hits",                    "GetNPartHits()"},   
		{ "phit_bar",          "bars in fPartHits",                         "fPartHits.SBSScintPartialHit.GetBarNum()"},
		{ "phit_case",         "Case number in fPartHits",                  "fPartHits.SBSScintPartialHit.GetCaseNum()"},
		{ "phit_lt",           "left tdc in fPartHits",                     "fPartHits.SBSScintPartialHit.GetLt()"},
		{ "phit_rt",           "right tdc in fPartHits",                    "fPartHits.SBSScintPartialHit.GetRt()"},
		{ "phit_la",           "left adc in fPartHits",                     "fPartHits.SBSScintPartialHit.GetLa()"},
		{ "phit_ra",           "right adc in fPartHits",                    "fPartHits.SBSScintPartialHit.GetRa()"},
		{ "phit_ltdc",         "left_raw tdc in fPartHits",                 "fPartHits.SBSScintPartialHit.GetLt_raw()"},
		{ "phit_rtdc",         "right_raw tdc in fPartHits",                "fPartHits.SBSScintPartialHit.GetRt_raw()"},
		{ "phit_ladc",         "left_raw adc in fPartHits",                 "fPartHits.SBSScintPartialHit.GetLa_raw()"},
		{ "phit_radc",         "right_raw adc in fPartHits",                "fPartHits.SBSScintPartialHit.GetRa_raw()"},
#endif
		// Reconstructed Multi-bar hits
		//{ "ncmbhit",           "Number of combined-bar hits",               "GetNCombinedHits()"},
		//{ "cmb_nbars",         "number of bars in combined hit",            "fCombHits.THaMultiHit.GetNBarHits()"},
		//{ "cmb_bar",           "primary bar of combined hit",               "fCombHits.THaMultiHit.GetBarNum()"},
		//{ "cmb_xpos",          "weighted X-position",                       "fCombHits.THaMultiHit.GetXpos()"},
		//{ "cmb_ypos",          "weighted Y-position",                       "fCombHits.THaMultiHit.GetYpos()"},
		//{ "cmb_dx2",           "weighted dx**2 of hit",                     "fCombHits.THaMultiHit.GetdX2()"},
		//{ "cmb_dy2",           "weighted dy**2 of hit",                     "fCombHits.THaMultiHit.GetdY2()"},
		//{ "cmb_tof",           "earliest TOF of hit--best bar",             "fCombHits.THaMultiHit.GetTof()"},
		//{ "cmb_Edep",          "total energy deposited",                    "fCombHits.THaMultiHit.GetTotEdep()"},

#if SAVEFLAT

		//bar with largetst ADC hit
		{ "MaxADCHitBar",         "Hit Bar number with max sqrt(RApedc*LApedc)",       "fMaxADCHitBar"},
		{ "MaxEnergyHitBar",         "Hit Bar number with max sqrt(RE*LE)",       "fMaxEnergyHitBar"},
		{ "MaxADCHit",         "max sqrt(RApedc*LApedc)",       "fMaxADCHit"},
		{ "MaxEnergyHit",         "max sqrt(RE*LE)",       "fMaxEnergyHit"},
		// copies of raw information but in flat arrays
		{ "LE",                "Left calibrated adc",                       "fLE"},
		{ "RE",                "Right calibrated adc",                      "fRE"},
		{ "LA",                "Left raw adc",                              "fLrawA"},
		{ "RA",                "Right raw adc",                             "fRrawA"},
		{ "LApedc",            "Left pedestal corrected adc",               "fLpedcA"},
		{ "RApedc",            "Right pedestal corrected adc",              "fRpedcA"},

		{ "LT",                "Left tdc value",                            "fLT"},
		{ "RT",                "Right tdc value",                           "fRT"},
		{ "LTcounter",         "Number of Hits on left tdc",                "fLTcounter"},
		{ "RTcounter",         "Number of Hits on right tdc",               "fRTcounter"},
		{ "hitcounter",        "Number of Hits",                            "hitcounter"},
		{ "Energy",            "Energy in bar",                             "Energy"},
		{ "Timediff",          "Time difference in bar",                    "TDIFF"},
		{ "tof",               "Time of flight",                            "TOF"},
		{ "Total_Time",        "Total Time",                                "T_tot"},
		{ "Yt_pos",            "Yt_pos",                                    "Yt_pos"},
		{ "Ya_pos",            "Ya_pos",                                    "Ya_pos"},
		{ "Y_pred",            "Y position prediction",                     "Y_pred"},
		{ "Y_dev",             "Deviation in Y actual and prediction",      "Y_dev"},
		{ "angle",             "angle (degrees) of bar hit from the origin","fAngle"},

#endif
		//track matching
		{"MatchRatioTrack","matched Hits / num of tracks",    "GetMatchRatioTrack()"},

		{ "trx",    "x-position of track in trigger plane",  "fTrackProj.THaTrackProj.fX" },
		{ "try",    "y-position of track in trigger plane",  "fTrackProj.THaTrackProj.fY" },
		{ "trpath", "TRCS pathlen of track to trigger plane","fTrackProj.THaTrackProj.fPathl" },
		{ "trdx",   "track deviation in x-position (m)", "fTrackProj.THaTrackProj.fdX" },
		{ "trHitIndex",  "the index of THaTriggerPlaneHit associated with track, -1 means no match",  "fTrackProj.THaTrackProj.fChannel" },



		{ 0 }
	};

	Int_t value = DefineVarsFromList( vars, mode );

	return value;


}



Int_t SBSTimingHodoscope::InitOutput( THaOutput* output )
{
	// Use the tree to store output




	if (fOKOut) return 0; // already initialized.
	if (!output) {
		Error("InitOutput","Cannot get THaOutput object. Output initialization FAILED!");
		return -2;
	}
	TTree* tree = output->GetTree();
	if (!tree) {
		Error("InitOutput","Cannot get Tree! Output initialization FAILED!");
		return -3;
	}


	//create the branches
	// if( tree->Branch(Form("%s.LT_0.",GetName()),"THaMtdc",fLT_0,15000) ) {
	//  fOKOut = true;
	//}

	fOKOut = true;

	if (fOKOut) return 0;
	return -1;
}


//_____________________________________________________________________________
SBSTimingHodoscope::~SBSTimingHodoscope()
{
	// Destructor. Remove variables from global list.

	if( fIsSetup )
		RemoveVariables();
	DeleteArrays();
}

//_____________________________________________________________________________
void SBSTimingHodoscope::DeleteArrays()
{
	// Delete member arrays. Used by destructor.

	delete fBars;
	delete fHits;
	//delete fCombHits;
	delete fLaHits;
	delete fRaHits;
	delete fLtHits;
	delete fRtHits;
	delete fRefHits;
	delete fRefCh;
	delete fPartHits;
	delete [] fLE;
	delete [] fRE;
	delete [] fLrawA;
	delete [] fRrawA;
	delete [] fLpedcA;
	delete [] fRpedcA;
	delete [] fLT;
	delete [] fRT;
	delete [] fLTcounter;
	delete [] fRTcounter;
	delete [] hitcounter;
	delete [] Energy;		
	delete [] TDIFF;
	delete [] T_tot;
	delete [] TOF;
	delete [] Yt_pos;
	delete [] Ya_pos;
	delete [] Y_pred;
	delete [] Y_dev;
	delete [] fAngle;
	delete [] fLtIndex;
	delete [] fRtIndex;
	delete [] fLaIndex;
	delete [] fRaIndex;

}

//_____________________________________________________________________________
inline 
void SBSTimingHodoscope::ClearEvent()
{
	// Reset per-event data.

#if 0
	const char cBig = 198;

	// quick set Int_t's to 0 or -1, and Double_t's to ~-1.e33
	memset(fLE, cBig, fNBars*sizeof(*fLE));
	memset(fRE, cBig, fNBars*sizeof(*fRE));
	memset(fLrawA, cBig,  fNBars*sizeof(*fLrawA));
	memset(fRrawA, cBig,  fNBars*sizeof(*fRrawA));
	memset(fLT, cBig,  fNBars*sizeof(*fLT));
	memset(fRT, cBig,  fNBars*sizeof(*fRT));
	memset(fLTcounter, 0, fNBars*sizeof(*fLTcounter));
	memset(fRTcounter, 0, fNBars*sizeof(*fRTcounter));
	memset(hitcounter, 0, fNBars*sizeof(*hitcounter));
	memset(fLpedcA, cBig, fNBars*sizeof(*fLpedcA));
	memset(fRpedcA, cBig, fNBars*sizeof(*fRpedcA));
	memset(Energy, cBig,  fNBars*sizeof(*Energy));
	memset(TDIFF, cBig,  fNBars*sizeof(*TDIFF));
	memset(TOF, cBig,  fNBars*sizeof(*TOF));
	memset(T_tot, cBig,  fNBars*sizeof(*T_tot));
	memset(Yt_pos, cBig,  fNBars*sizeof(*Yt_pos));
	memset(Ya_pos, cBig,  fNBars*sizeof(*Ya_pos));
	memset(Y_pred, cBig,  fNBars*sizeof(*Y_pred));
	memset(Y_dev, cBig,  fNBars*sizeof(*Y_dev));
	memset(fAngle, cBig,  fNBars*sizeof(*fAngle));
	memset(fLtIndex, 255,  fNBars*sizeof(*fLtIndex));
	memset(fRtIndex, 255,  fNBars*sizeof(*fRtIndex));
	memset(fLaIndex, 255,  fNBars*sizeof(*fLaIndex));
	memset(fRaIndex, 255,  fNBars*sizeof(*fRaIndex));

#else   
	const Double_t Big=-1.e35;

	for(Int_t i=0; i<fNBars; i++){
		fLE[i]=Big;
		fRE[i]=Big;
		fLrawA[i]=Big;
		fRrawA[i]=Big;
		fLT[i]=Big;
		fRT[i]=Big;
		fLTcounter[i]=0;
		fRTcounter[i]=0;
		hitcounter[i] = 0;
		fLpedcA[i]=Big;
		fRpedcA[i]=Big;


		Energy[i] = Big;		
		TDIFF[i] = Big;
		TOF[i] = Big;
		T_tot[i] =  Big;
		Yt_pos[i]  = Big;
		Ya_pos[i]  = Big;
		Y_pred[i] =  Big*.5;
		Y_dev[i] = Big;
		fAngle[i] = Big;

		fLtIndex[i] = -1;
		fRtIndex[i] = -1;
		fLaIndex[i] = -1;
		fRaIndex[i] = -1;
	}
#endif

	fMaxADCHitBar=fMaxEnergyHitBar=-1; //set invalid

	fLTNhit = 0;                       // Number of Left paddles TDC times
	fRTNhit = 0;                       // Number of Right paddles TDC times
	fLANhit = 0;                       // Number of Left paddles ADC amps
	fRANhit = 0;                       // Number of Right paddles ADC amps
	fLaHits->Clear("C");
	fRaHits->Clear("C");
	fLtHits->Clear("C");
	fRtHits->Clear("C");
	fHits->Clear("C");
	//fCombHits->Clear("C");
	fPartHits->Clear();
	fRefHits->Clear("C");
	fRefOkay = true ;


	fCoarseProcessed = kFALSE;
	fFineProcessed = kFALSE;

}

//_____________________________________________________________________________
Double_t SBSTimingHodoscope::TimeWalkCorrection( SBSScintPMT* pmt, Double_t ADC, Double_t time)
{
	// Calculate the time-walk correction. The timewalk might be
	// dependent upon the specific PMT, so information about exactly
	// which PMT fired is required.
	Double_t adc=ADC;

	// get the ADC value above the pedestal
	if ( adc <=0. ) return time;

	Double_t ref = 1.e10;   // a constant of "infinity" in this case such that tw_ref=0
	Double_t tw = 0.0;
	Double_t tw_ref = 0.0;
	Double_t par  = pmt->GetTimeWalk();
	Double_t wexp = pmt->GetTimeWExp();

	tw = par*pow(adc,wexp);
	tw_ref = par*pow(ref,wexp);

	return time - tw + tw_ref;
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::Decode( const THaEvData& evdata )
{
	// Decode scintillator data, correct TDC times and ADC amplitudes, and copy
	// the data to the local data members.
	static const char *here="Decode()";

	ClearEvent();

	fEventCount++;

	if (!evdata.IsPhysicsTrigger()) return -1;

#if DEBUG_LEVEL>=4//massive info
	cout <<"-- SBSTimingHodoscope (name: "<<GetName()<<") Decode THaEvData of Run#"<<evdata.GetRunNum()<<" --"<<endl;
	cout<<"EvType="<<evdata.GetEvType()<<"; EvLength="<<evdata.GetEvLength()<<"; EvNum="<<evdata.GetEvNum()<<endl;
#endif//#if DEBUG_LEVEL>=4

#if DEBUG_LEVEL>=5//super massive info
	cout<<">>>> search detector map in SBSTimingHodoscope::Decode"<<endl;
	Int_t tmpidx=-1;
	while (++tmpidx < fDetMap->GetSize()) {
		THaDetMap::Module * d = fDetMap->GetModule(tmpidx);

		// Get number of channels with hits
		for (Int_t chan=d->lo;chan<=d->hi;chan++){
			Int_t nHits = evdata.GetNumHits(d->crate, d->slot, chan);
			if (nHits)
				cout<<"d->crate="<<d->crate<<" , d->slot="<< d->slot<<" , chan="<<chan
				<<" nHits="<<nHits<<endl;
		}
	}
	cout <<">>>>End search"<<endl;
#endif

	Int_t nextLaHit=0;
	Int_t nextRaHit=0;
	Int_t nextLtHit=0;
	Int_t nextRtHit=0;

	// Loop over all detector modules containing reference lines
	// they have to be at the beginning of the detector map
	// and we need one line (=one module) per reference channel
	// otherwise only the first will be used
	Int_t i=0;
	Int_t data;
	while ((i < fDetMap->GetSize())&&(  i < GetNRefCh() )) {
		THaDetMap::Module * d = fDetMap->GetModule(i);

		// Get number of channels with hits
		Int_t chan=d->lo;
		Int_t nHits = evdata.GetNumHits(d->crate, d->slot, chan);

#if DEBUG_THIS
		cout << "Found " << nHits << "  hits in the reference channel for " << GetName()
			<< " at " << d->crate << "  " << d->slot << "  " << chan << endl;
#endif

		//complaining about error reference channel
		if (nHits!=1) 
			fErrorReferenceChCount++;

		fTooManyErrRefCh=
			((Double_t)fErrorReferenceChCount/(Double_t)fEventCount>
			fErrorReferenceChRateWarningThreshold)
			and
			fEventCount>2000;
		;

		if (nHits!=1 and fTooManyErrRefCh) 
		{
                    /*
			DEBUG_WARNING(Here(here),
				"Till Event %d, There are too many events with error"
				"reference channel(%f\%)."
				"The last error states as following:",
				fEventCount,
				100.*(Double_t)fErrorReferenceChCount/(Double_t)fEventCount
				);
                                */
		}

		if (nHits<1) {
#if DEBUG_LEVEL>=2//this warning occurs from time to time, lower its level to info.
			if (fTooManyErrRefCh)
			{
				stringstream s;
				s<<"\tEvent Number "
					<<evdata.GetEvNum()
					<<"; NO Hits for Reference Channel, ignore this event"
					<<i<<" of detector"<<fPrefix<<endl;
				Warning(Here(here),"%s",s.str().c_str());
			}
#endif//#if DEBUG_LEVEL>=2
			data = 2^31 ;//new error value to 
			return (0);//Jin Huang
		} else {
			if (nHits>1) {
#if DEBUG_LEVEL>=2//start show info
				if (fTooManyErrRefCh)
				{
					stringstream s;
					s<<"\tEvent Number "
						<<evdata.GetEvNum()
						<<"; Multiple Hits for Reference Channel "<<i
						<<" of detector"<<fPrefix
						<<"Using first one"<<endl;
					Warning(Here(here),"%s",s.str().c_str());
				}
#endif//#if DEBUG_LEVEL>=3
			}
			data = evdata.GetData(d->crate, d->slot, chan, 0);
		}

#if DEBUG_THIS
		cout<<"Use data=" << data << endl;
#endif
		// PMT numbers and channels go in the same order ... 
		const SBSScintPMT* pmt = GetRefCh(i);
		if( !pmt ) { 
			cout<<"SBSTimingHodoscope::Decode : Ref Channels are not initialized"<<endl;
			cout<<"Skipping event"<<endl;
			return -1;
		}
		new( (*fRefHits)[i] )  SBSTdcHit( pmt , data );    
		i++;
	}
	if (i!=GetNRefCh()) {
		cout<<"SBSTimingHodoscope::Decode : Mismatch between fNRefCh and hits on RefLines"<<endl;
		cout<<i<<" "<<GetNRefCh()<<endl;
		return -1;
	}

	Int_t nref = GetNRefCh();

	// two ways to process the data -- one is good for densely packed data in only
	// one module, the second better for more scattered out data

	while (i < fDetMap->GetSize()){
		THaDetMap::Module * d = fDetMap->GetModule(i);
		Bool_t isAdc=fDetMap->IsADC(d);

		Bool_t known = fDetMap->IsADC(d) || fDetMap->IsTDC(d);
#if DEBUG_LEVEL>=1//start show warning 
		if(!known) Warning(Here(here),
			"\tUnknown Module %d @ crate=%d, slot=%d",i,d->crate ,d->slot);
#endif//#if DEBUG_LEVEL>=1

		// NOT IMPLEMENTED YET:
		// for speed, there is another version that creates a look-up
		// table for each module as to the appropriate channel number

#if 0    /* Good for compact cabling */
		// Get number of channels with hits
		Int_t nChan = evdata.GetNumChan(d->crate, d->slot);

		for (Int_t chNdx = 0; chNdx < nChan; chNdx++) {
			// Use channel index to loop through channels that have hits

			Int_t chan = evdata.GetNextChan(d->crate, d->slot, chNdx);
			if (chan < d->lo || chan > d->hi) 
				continue; //Not part of this detector
#else         /* Better for sparse cabling */
		for (Int_t chan=d->lo; chan<=d->hi; chan++) {
#endif

			DEBUG_MASSINFO(Here(here),
				"Trying d->crate=%d, d->slot=%d, chan=%d. Slot info:",
				d->crate, d->slot, chan);
			DEBUG_LINE_MASSINFO(evdata.PrintSlotData(d->crate, d->slot));

			// Get number of hits for this channel and loop through hits
			Int_t nHits = evdata.GetNumHits(d->crate, d->slot, chan);

			if (nHits<=0) continue;
#if DEBUG_THIS
			if (fDetMap->IsTDC(d))
				cout<<"Decode found "<<nHits<<" TDC hit(s) @ crate="<<d->crate
				<<", slot="<<d->slot<<", chan="<<chan<<endl;
			else
				cout<<"Decode found "<<nHits<<" hit(s) @ crate="<<d->crate
				<<", slot="<<d->slot<<", chan="<<chan<<endl;
#endif			


			Bool_t isLeft;
			SBSScintBar* abar=NULL;
			SBSScintPMT* pmt=NULL;

			// PMT  numbers and channels go in the same order ... 
			// PMTNum is kind of artificial, since it comes from the 
			// way the detector map is read in: so the first GetNbars are 
			// left TDC channels, the next right TDC channels and so on
			Int_t PMTNum  = d->first + chan - d->lo;

			// due to numbering, the first channels are timing reference channels,
			// so remove them
			PMTNum -= nref;

			Int_t nbars = GetNBars();

			isLeft = (PMTNum/nbars % 2)==0;

			abar=GetBar( PMTNum % GetNBars() );

#if DEBUG_LEVEL>=1//start show warning 
			if (!abar) Warning(Here(here),"\tcan not get bar #%d",PMTNum % GetNBars() );
#endif//#if DEBUG_LEVEL>=1

			// in case the readout-module type is unknown, fall back to numbering scheme
			if ( isAdc || (!known && PMTNum>=2*GetNBars()) ) {
				// ADC
				isAdc=kTRUE;
			} else {
				// TDC
				isAdc=kFALSE;
			}

			if (!abar) {
				continue;
			}
			if (isLeft) { pmt = abar->GetLPMT(); } else { pmt = abar->GetRPMT(); }

#if DEBUG_LEVEL>=1//start show warning 
			if (!pmt) Warning(Here(here),"\tcan not get pmt on bar #%d",PMTNum % GetNBars() );
#endif//#if DEBUG_LEVEL>=1
			if( !pmt ) { continue;}

#if DEBUG_LEVEL>=5 //hardware check for #7
			if (!isAdc&&(PMTNum % GetNBars()==6||PMTNum % GetNBars()==7||PMTNum % GetNBars()==8))
			{
				if (!isAdc&&isLeft&&PMTNum % GetNBars()==6) 
					Info(Here(here),"\tEvent #%d get TDC hit on left bar #%d",evdata.GetEvNum(),PMTNum % GetNBars() );
				if (!isAdc&&!isLeft&&PMTNum % GetNBars()==6) 
					Info(Here(here),"\tEvent #%d get TDC hit on right bar #%d",evdata.GetEvNum(),PMTNum % GetNBars() );
				if (!isAdc&&isLeft&&PMTNum % GetNBars()==7) 
					Info(Here(here),"\tEvent #%d get TDC hit on left bar #%d",evdata.GetEvNum(),PMTNum % GetNBars() );
				if (!isAdc&&!isLeft&&PMTNum % GetNBars()==7) 
					Info(Here(here),"\tEvent #%d get TDC hit on right bar #%d",evdata.GetEvNum(),PMTNum % GetNBars() );
				if (!isAdc&&isLeft&&PMTNum % GetNBars()==8) 
					Info(Here(here),"\tEvent #%d get TDC hit on left bar #%d",evdata.GetEvNum(),PMTNum % GetNBars() );
				if (!isAdc&&!isLeft&&PMTNum % GetNBars()==8) 
					Info(Here(here),"\tEvent #%d get TDC hit on right bar #%d",evdata.GetEvNum(),PMTNum % GetNBars() );
			}
#endif

			// loop through the hits
			for (Int_t hit = 0; hit < nHits; hit++) {
				// Loop through all hits for this channel, and store the
				// TDC/ADC  data for this hit
				Int_t data = evdata.GetData(d->crate, d->slot, chan, hit);
				if (isAdc) {  // it is an ADC module
					if (isLeft) {              
						new( (*fLaHits)[nextLaHit++] )  SBSAdcHit( pmt, data );
					} else {
						new( (*fRaHits)[nextRaHit++] )  SBSAdcHit( pmt, data );
					}
				}
				else { // it is a TDC module and hit
					Double_t timeoff = 0.;
					if ((d->refindex)>=0) {
						// handle the clock/scaler based TDCs and remove the reference time,
						// taking care of when the clock rolls-over
						const SBSTdcHit* ahit = GetRefHit(d->refindex);
						if (ahit && ahit->GetRawTime()>=0) {
							Int_t diff = data - ahit->GetRawTime();
							Double_t wrap = pmt->GetRawWrapAround();
							if ( diff < -wrap/2. ) {
								timeoff = ahit->GetTime() -  wrap * pmt->GetTDCRes() ;
							} else if ( diff > wrap/2. ) {
								timeoff = ahit->GetTime() +  wrap * pmt->GetTDCRes() ;
							} else {
								timeoff = ahit->GetTime();
							}
						}
						else {  // missing reference-timing hit
							fRefOkay = false ;
						}

					}

					if (isLeft) {              
						new( (*fLtHits)[nextLtHit++] )  SBSTdcHit( pmt, data, timeoff );
					} else {
						new( (*fRtHits)[nextRtHit++] )  SBSTdcHit( pmt, data, timeoff );
					}
				}
			} // End hit loop
		} // End channel index loop
		i++;
	}

	// copy into the flat arrays for easy viewing
#if SAVEFLAT
	Int_t barno=0;
	for(Int_t i=0; i< nextLaHit; i++){
		const SBSAdcHit *la = GetLaHit(i);
		barno=la->GetPMT()->GetBarNum();
		fLrawA[barno]=la->GetRawAmpl();    
		fLpedcA[barno]=la->GetAmplPedCor();    
		fLE[barno]=la->GetAmpl();    
	}

	for(Int_t i=0; i< nextRaHit; i++){
		const SBSAdcHit *ra = GetRaHit(i);
		barno=ra->GetPMT()->GetBarNum();
		fRrawA[barno]=ra->GetRawAmpl();    
		fRpedcA[barno]=ra->GetAmplPedCor();    
		fRE[barno]=ra->GetAmpl();    
	}

	for(Int_t i=0; i< nextLtHit; i++){
		const SBSTdcHit *lt = GetLtHit(i);
		barno=lt->GetPMT()->GetBarNum();
		fLTcounter[barno]++;
		if (fLTcounter[barno]==1) {
			fLT[barno]=lt->GetTime();
		}
	}   

	for(Int_t i=0; i< nextRtHit; i++){
		const SBSTdcHit *rt = GetRtHit(i);
		barno=rt->GetPMT()->GetBarNum();
		fRTcounter[barno]++;
		if (fRTcounter[barno]==1) {
			fRT[barno]=rt->GetTime();
		}
	}

	Double_t MaxEnergy=0, MaxADC=0;
	for (barno=0;barno<GetNBars();barno++)
	{
		if (fLpedcA[barno]>0 && fLE[barno]>0 && fRpedcA[barno]>0 && fRE[barno]>0)
		{
			if (fLpedcA[barno]*fRpedcA[barno]>MaxADC) 
			{fMaxADCHitBar=barno; MaxADC=fLpedcA[barno]*fRpedcA[barno];}
			if (fLE[barno]*fRE[barno]>MaxEnergy) 
			{fMaxEnergyHitBar=barno; MaxEnergy=fLE[barno]*fRE[barno];}
		}
	}
	fMaxADCHit = sqrt(MaxADC);
	fMaxEnergyHit = sqrt(MaxEnergy);

#endif


	return nextLtHit + nextRtHit + nextLaHit + nextRaHit;
}


//_____________________________________________________________________________
Int_t SBSTimingHodoscope::CoarseProcess( TClonesArray& tracks )
{
	if (fCoarseProcessed) return 0;

	//cout<<"Entering ScintPlane CourseProcess "<<endl;

	// sort hits by bar number, and then by value, earliest/highest amplitude first
	fLtHits->Sort();
	fRtHits->Sort();
	fLaHits->Sort();
	fRaHits->Sort();

	Int_t nlt = GetNLtHits();
	Int_t nrt = GetNRtHits();
	Int_t nla = GetNLaHits();
	Int_t nra = GetNRaHits();

	// build the indices
	for (Int_t i=0; i< nlt; i++) {
		Int_t barno=static_cast<SBSTdcHit*>(fLtHits->At(i))->GetPMT()->GetBarNum();
		if (fLtIndex[barno]<0) fLtIndex[barno] = i;
	}
	for (Int_t i=0; i< nrt; i++) {
		Int_t barno=static_cast<SBSTdcHit*>(fRtHits->At(i))->GetPMT()->GetBarNum();
		if (fRtIndex[barno]<0) fRtIndex[barno] = i;
	}
	for (Int_t i=0; i< nla; i++) {
		Int_t barno=static_cast<SBSAdcHit*>(fLaHits->At(i))->GetPMT()->GetBarNum();
		if (fLaIndex[barno]<0) fLaIndex[barno] = i;
	}
	for (Int_t i=0; i< nra; i++) {
		Int_t barno=static_cast<SBSAdcHit*>(fRaHits->At(i))->GetPMT()->GetBarNum();
		if (fRaIndex[barno]<0) fRaIndex[barno] = i;
	}

	fCoarseProcessed=kTRUE;

#if BUILD_PARTIAL_HIT      
	return BuildAllBars(tracks);
#else
	Int_t nhits = BuildCompleteBars(tracks);/* new version */
	//if (nhits>0) CombineHits(tracks); //only for neutron detector setup
	return nhits;
#endif
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::BuildAllBars( TClonesArray& tracks )
{
	// go through ALL the hits and build not only the complete hits, but also
	// the partial hits

	Int_t HitNum=0;
	Int_t PartHitNum=0;
	Int_t CaseNum=0;
	SBSScintBar* ptBar;
	Double_t Lt_c=0.0;
	Double_t Rt_c=0.0;
	Double_t La_c=0.0;
	Double_t Ra_c=0.0;
	Double_t Lt_raw=0.0;
	Double_t Rt_raw=0.0;
	Double_t La_raw=0.0;
	Double_t Ra_raw=0.0;


	Double_t raddeg =180.0/TMath::Pi();
	Double_t bx=0.0;
	Double_t bz=0.0;
	Double_t Xax_an=0.0;


	Double_t Ypos_c=0.0;
	Double_t Ya=0.0;
	Double_t Ampl=0.0;
	Double_t Tdiff=0.0;
	Double_t Tof=0.0;
	Double_t LeftTime=0.0;
	Double_t RightTime=0.0;

	// idea: for each TDC fire, construct a full hit from the ADC and other side TDC
	//      since multi-hit TDCs, right now constructing a set for each combination
	//
	//       extra hits (out-of-time) will be culled via cuts placed in the database

	//     For complete information, build a real 'Hit' into fHits
	//       otherwise, build something less useful into fPartialHits

	const SBSTdcHit *lt, *rt;
	const SBSAdcHit *la, *ra;
	for(Int_t BarNum=0; BarNum<fNBars; BarNum++) {
		ptBar = GetBar(BarNum);
		la = GetBarHitA('l',ptBar);
		ra = GetBarHitA('r',ptBar);

		Double_t cn     = ptBar->GetC();
		Double_t attLen = ptBar->GetAtt();
		//Double_t ybar   = ptBar->GetYWidth();

		rt = 0;
		Int_t ilt;
		for(ilt=0; (lt = GetBarHitT('l',ptBar,ilt)); ilt++) {
			Int_t irt;
			for (irt=0; (rt = GetBarHitT('r',ptBar,irt)); irt++) {
				// loop through timed hits
				rt = GetBarHitT('r',ptBar,0);

				if (lt && rt && la && ra) { // complete hit
					Lt_c = lt->GetTime();
					Rt_c = rt->GetTime();
					// Only use ped-subtracted for time-walk
					La_c = la->GetAmplPedCor();
					Ra_c = ra->GetAmplPedCor();

					Ya = TMath::Log(La_c/Ra_c)*attLen/2.0;
					//Ampl = TMath::Sqrt(La_c*Ra_c*TMath::Exp(fAttLen*2*fYBar));
					//Ampl = TMath::Sqrt(La_c*Ra_c*TMath::Exp(fYBar/fAttLen));
					Ampl = TMath::Sqrt(La_c*Ra_c);
					LeftTime = TimeWalkCorrection( lt->GetPMT(), La_c, Lt_c );
					RightTime= TimeWalkCorrection( rt->GetPMT(), Ra_c, Rt_c );
					//Tdiff=RightTime-LeftTime;
					Tdiff= .5*(Rt_c - Lt_c);
					//Tof = .5*(LeftTime + RightTime)- fYBar/fCn;
					//	  Tof = .5*(Lt_c + Rt_c)-ybar/(2*cn);
					Tof = .5*(Lt_c + Rt_c);
					//Ypos_c = .5*fCn*Tdiff;
					Ypos_c = .5*cn*(Rt_c - Lt_c);


					// what is trying to be done here is very unclear	  
					Xax_an= TMath::ATan2(fXax.X(),fXax.Z());

					bx = fOrigin.X() +  Ypos_c*sin(Xax_an);
					bz = fOrigin.Z() +  Ypos_c*cos(Xax_an);
					//P2_an = TMath::ATan2(P2_x,P2_z);
					//P2_an *= raddeg;


					if (hitcounter[BarNum]==0) {
						Energy[BarNum] = Ampl;		
						TDIFF[BarNum]  = Tdiff;
						T_tot[BarNum]  = Lt_c + Rt_c;
						TOF[BarNum]    = Tof;
						Yt_pos[BarNum] = Ypos_c;
						Ya_pos[BarNum] = Ya;
						fAngle[BarNum] = TMath::ATan2(bx,bz)*raddeg;
						hitcounter[BarNum]++;
					}

					// Not applying any consistency-cut at this level	
					new( (*fHits)[HitNum] )  SBSScintHit(  ptBar ,0,BarNum, Ypos_c, Tof, Ampl, Tdiff);   // add bar num
					HitNum++;		    
				}  
				else if (lt && rt && la) {
					// partial hit with lt,rt,la but with no ra (case 14)

					Lt_c = lt->GetTime();
					Rt_c = lt->GetTime();
					La_c = la->GetAmpl();
					Lt_raw = lt->GetRawTime();
					Rt_raw = rt->GetRawTime();
					La_raw = la->GetRawAmpl();

					CaseNum = 14;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum, Lt_c, Lt_raw, Rt_c , Rt_raw,
						La_c, La_raw, 0.0,0.0 );  
					PartHitNum++;	      
				}
				else if (lt && rt && ra) {
					//partial hit with lt, rt and ra but no la (case 13)
					Lt_c = lt->GetTime();
					Rt_c = rt->GetTime();
					Ra_c = ra->GetAmpl();

					Lt_raw = lt->GetRawTime();
					Rt_raw = rt->GetRawTime();
					Ra_raw = ra->GetRawAmpl();

					CaseNum = 13;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar ,BarNum, CaseNum, Lt_c,Lt_raw, Rt_c , Rt_raw ,
						0.0 , 0.0, Ra_c, Ra_raw );  
					PartHitNum++;
				}
				else if (lt && rt) {
					//partial hit with lt, rt but no la,ra (case 12)
					Lt_c = lt->GetTime();
					Rt_c = rt->GetTime();

					Lt_raw = lt->GetRawTime();
					Rt_raw = rt->GetRawTime();

					CaseNum = 12;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar ,BarNum, CaseNum, Lt_c, Lt_raw ,Rt_c , Rt_raw,
						0.0 ,0.0 ,0.0, 0.0 );  
					PartHitNum++;

				}
			}
			if (irt==0) { // no rt found
				if (lt && la && ra) {
					// partial hit with lt,la,ra but no rt (case 11)	      	
					Lt_c = lt->GetTime();
					La_c = la->GetAmpl();
					Ra_c = ra->GetAmpl();

					Lt_raw = lt->GetRawTime();
					La_raw = la->GetRawAmpl();
					Ra_raw = ra->GetRawAmpl();

					CaseNum = 11;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar ,BarNum, CaseNum, Lt_c, Lt_raw , 0.0, 0.0 ,
						La_c, La_raw , Ra_c,Ra_raw );  
					PartHitNum++;
				} else if (lt && la) {
					// partial hit with lt,la but with no rt,ra (case 10)

					Lt_c = lt->GetTime();
					La_c = la->GetAmpl();

					Lt_raw = lt->GetRawTime();
					La_raw = la->GetRawAmpl();

					CaseNum = 10;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum, Lt_c, Lt_raw, 0.0, 0.0 ,
						La_c, La_raw ,0.0, 0.0  );  
					PartHitNum++;
				} else if (lt && ra) {
					//partial hit with lt and ra but no la,rt (case 9)
					Lt_c = lt->GetTime();
					Ra_c = ra->GetAmpl();

					Lt_raw = lt->GetRawTime();
					Ra_raw = ra->GetRawAmpl();

					CaseNum = 9;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum, Lt_c,Lt_raw ,0.0 , 0.0,
						0.0 ,0.0, Ra_c, Ra_raw );  
					PartHitNum++;
				} else if (lt) {
					//partial hit with lt but no la,ra,rt (case 8)
					Lt_c = lt->GetTime();
					Lt_raw = lt->GetRawTime();

					CaseNum = 8;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum, Lt_c,Lt_raw ,0.0, 0.0,
						0.0 , 0.0, 0.0, 0.0 );  
					PartHitNum++;
				}
			}
		}
		if (ilt==0) { // no lt found
			Int_t irt;
			for (irt=0; (rt = GetBarHitT('r',ptBar,irt)); irt++) {
				// loop through timed hits
				rt = GetBarHitT('r',ptBar,0);
				if (rt && ra && la) {
					// partial hit with rt,la,ra but no lt (case 7)

					Rt_c = rt->GetTime();
					La_c = la->GetAmpl();
					Ra_c = ra->GetAmpl();

					Rt_raw = rt->GetRawTime();
					La_raw = la->GetRawAmpl();
					Ra_raw = ra->GetRawAmpl();

					CaseNum = 7;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum,0.0 ,0.0, Rt_c , Rt_raw,
						La_c, La_raw , Ra_c, Ra_raw );  
					PartHitNum++;
				} else if (rt && la) {
					// partial hit with rt,la but with no lt,ra (case 6)

					Rt_c = rt->GetTime();
					La_c = la->GetAmpl();

					Rt_raw = rt->GetRawTime();
					La_raw = la->GetRawAmpl();

					CaseNum = 6;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum, 0.0 ,0.0, Rt_c ,Rt_raw ,
						La_c, La_raw, 0.0, 0.0 );  
					PartHitNum++;
				} else if (rt && ra) {
					//partial hit with rt and ra but no la, lt (case 5)

					Rt_c = rt->GetTime();
					Ra_c = ra->GetAmpl();

					Rt_raw = rt->GetRawTime();
					Ra_raw = ra->GetRawAmpl();

					CaseNum = 5;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum,0.0, 0.0 , Rt_c , Rt_raw,
						0.0 , 0.0, Ra_c, Ra_raw );  
					PartHitNum++;
				} else if (rt) {
					//partial hit with rt but no la,ra,lt (case 4)

					Rt_c = rt->GetTime();

					Rt_raw = rt->GetRawTime();

					CaseNum = 4;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit(  ptBar,BarNum, CaseNum,0.0 , 0.0, Rt_c, Rt_raw ,
						0.0 , 0.0, 0.0, 0.0  );  
					PartHitNum++;
				}
			}
			if (irt==0 && ilt==0) {
				// no timing hits whatsoever
				if (la&&ra) {
					La_c = la->GetAmpl();
					Ra_c = ra->GetAmpl();

					La_raw = la->GetRawAmpl();
					Ra_raw = ra->GetRawAmpl();

					CaseNum = 3;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum,0.0 ,0.0, 0.0 , 0.0,
						La_c, La_raw, Ra_c, Ra_raw);  
					PartHitNum++;
				} else if (la) {
					// partial hit including la   but no lt ,rt and ra (case 2)
					La_c = la->GetAmpl();

					La_raw = la->GetRawAmpl();

					CaseNum = 2;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit(  ptBar,BarNum, CaseNum,0.0 , 0.0, 0.0, 0.0,
						La_c, La_raw, 0.0, 0.0 );  
					PartHitNum++;
				} else if (ra) {
					// partial hit including ra   but no la, lt , rt (case 1)

					Ra_c = ra->GetAmpl();

					Ra_raw = ra->GetRawAmpl();

					CaseNum = 1;
					new( (*fPartHits)[PartHitNum] )  SBSScintPartialHit( ptBar,BarNum, CaseNum, 0.0, 0.0, 0.0, 0.0,
						0.0, 0.0, Ra_c, Ra_raw );  
					PartHitNum++;
				}
			}
		}
		// below is vestigial -- and I do not believe it produces anything interesting (RJF)
		//#if 0    
		//		if (hitcounter[BarNum]) {
		//			Double_t M,C;
		//			Int_t i=BarNum;
		//			M = 1.e5;
		//			C = 1.e5;
		//			if(i==0){
		//				M = (Yt_pos[2] - Yt_pos[1])/( GetBar(2)->GetXPos() -  GetBar(1)->GetXPos()  );
		//				C = Yt_pos[1] - M*GetBar(1)->GetXPos();
		//			}
		//			else{
		//				if(i==GetNBars()-1){
		//					M = (Yt_pos[i-1] - Yt_pos[i-2])/( GetBar(i-1)->GetXPos() -  GetBar(i-2)->GetXPos()  );
		//					C = Yt_pos[i-1] - M*GetBar(i-1)->GetXPos();
		//				}else{
		//					M = (Yt_pos[i+1] - Yt_pos[i-1])/( GetBar(i+1)->GetXPos() -  GetBar(i-1)->GetXPos()  );
		//					C = Yt_pos[i+1] - M*GetBar(i+1)->GetXPos();
		//				}
		//			}
		//
		//			Y_pred[i] = M*GetBar(i)->GetXPos() + C;
		//			Y_dev[i] = Yt_pos[i] -  Y_pred[i];
		//
		//		}
		//#endif
	}
#if DEBUG_THIS
	assert(GetNHits()==HitNum);
	assert(GetNPartHits()==PartHitNum);
	cout<<"Reconstruct summery for "<<GetName()<<" Subdetector: "<<endl
		<<"\tGetNRefHits="<<GetNRefHits()<<endl
		<<"\tGetNLtHits="<<GetNLtHits()<<'\t'
		<<"\tGetNRtHits="<<GetNRtHits()<<endl
		<<"\tGetNLaHits="<<GetNLaHits()<<'\t'
		<<"\tGetNRaHits="<<GetNRaHits()<<endl
		<<"from which I build "<<HitNum<<" hit(s) and "<<PartHitNum<<" partial hit(s) with BuildAllBars()"<<endl;
#endif
	return HitNum; 

}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::FineProcess( TClonesArray& tracks )
{
	//TODO: add fine proc code

	if (fFineProcessed) return 0;
	fFineProcessed=kTRUE;

	return FineMatchingHits(tracks);
}

//_____________________________________________________________________________
const SBSTdcHit* SBSTimingHodoscope::GetBarHitT(const char side,
											  const SBSScintBar *const ptr,
											  const int n) const 
{
	// return matching Tdc from bar ptr on side, the n'th Tdc signal
	const TClonesArray *arr;
	const Int_t *index;
	if (side=='L' || side=='l') {
		arr=fLtHits;
		index=fLtIndex;
	} else {
		arr=fRtHits;
		index=fRtIndex;
	}

	const SBSTdcHit *r=0;

	Int_t barno = ptr->GetBarNum();
	if (barno>=0 && index[barno]>=0) {
		int cnt=0;
		int narr = arr->GetLast();
		for (int i=index[barno]; i<=narr; i++) {
			const SBSTdcHit *h = static_cast<const SBSTdcHit*>(arr->At(i));
			if ( h->GetPMT()->GetScintBar() == ptr ) {
				if ( cnt == n ) {
					r = h;
					break;
				} else {
					cnt++;
				}
			} else {
				break; // passed this bar's hits
			}
		}
	}
	return r;
}


//_____________________________________________________________________________
const SBSAdcHit* SBSTimingHodoscope::GetBarHitA(const char side,
											  const SBSScintBar *const ptr,
											  const int n) const
{
	// return matching Adc from bar ptr on side, return n'th signal (over-doingit)
	TClonesArray *arr;
	const Int_t *index;
	if (side=='L' || side=='l') {
		arr=fLaHits;
		index=fLaIndex;
	} else {
		arr=fRaHits;
		index=fRaIndex;
	}

	SBSAdcHit *r=0;

	Int_t barno = ptr->GetBarNum();
	if (barno>=0 && index[barno]>=0) {
		int cnt=0;
		int narr = arr->GetLast();
		for (int i=index[barno]; i<=narr; i++) {
			SBSAdcHit *h = static_cast<SBSAdcHit*>(arr->At(i));
			if ( h->GetPMT()->GetScintBar() == ptr ) {
				if ( cnt == n ) {
					r = h;
					break;
				}
			} else {
				break; // passed this bar's hits
			}
		}
	}
	return r;
}

//_____________________________________________________________________________
Int_t SBSTimingHodoscope::BuildCompleteBars( TClonesArray& tracks ) {
	// idea: for each TDC fire, construct a full hit from the ADC and other side TDC
	//      since multi-hit TDCs, right now constructing a set for each combination
	DEBUG_LINE_INFO(static const char *here="BuildCompleteBars");

	const Double_t Big=-1.e35;

	Int_t HitNum=0;

	// loop through Left-TDCs first, only worrying about complete hits

	SBSScintBar* ptBar;
	Double_t yt, tof, amp, tdiff;
	Int_t nlt = GetNLtHits();
	for (Int_t i=0; i<nlt; i++) {  // loop through left-hits
		const SBSTdcHit *lt = GetLtHit(i);
		ptBar = lt->GetPMT()->GetScintBar();

		// make sure the hit is within "range"
		{
			Double_t t = lt->GetTime();
			SBSScintPMT *pmt = lt->GetPMT();
			if (t < pmt->GetRawLowLim() || t > pmt->GetRawUpLim() )
			{
#if DEBUG_LEVEL>=3// info
				Info(Here(here),"\tHit time (t=%d) out of range, ignore this hit", t);
#endif//#if DEBUG_LEVEL>=3
				lt=0;
			}
		}

		// ADC info
		const SBSAdcHit *la = GetBarHitA('l',ptBar);
		const SBSAdcHit *ra = GetBarHitA('r',ptBar);

		const SBSTdcHit *rt=0;
		int rcnt=0;
		for (rcnt=0; ( rt=GetBarHitT('r',ptBar,rcnt) ); rcnt++) {
			{
				Double_t t = rt->GetTime();
				SBSScintPMT *pmt = rt->GetPMT();
				if (t < pmt->GetRawLowLim() || t > pmt->GetRawUpLim() )
				{
#if DEBUG_LEVEL>=3// info
					Info(Here(here),"\tHit time (t=%d) out of range, ignore this hit", t);
#endif//#if DEBUG_LEVEL>=3
					rt=0;
				}
			}
			if (lt && la && rt && ra) {
				// we now have a complete set!
				// build the paddle hit
				Double_t cn  = ptBar->GetC();
				Double_t att = ptBar->GetAtt();
				//Double_t len = ptBar->GetYWidth();

				Int_t bar = ptBar->GetBarNum();
				Double_t ltime = TimeWalkCorrection( lt->GetPMT(), la->GetAmplPedCor(), lt->GetTime() );
				Double_t rtime = TimeWalkCorrection( rt->GetPMT(), ra->GetAmplPedCor(), rt->GetTime() );

				tdiff = 0.5*(rtime - ltime);
				//	tof   = 0.5*(rtime + ltime) - .5*len/cn;
				tof   = 0.5*(rtime + ltime);
				yt    = tdiff*cn;

				Double_t lamp = la->GetAmpl();
				Double_t ramp = ra->GetAmpl();
				Double_t ya=Big;
				amp = 0;

				if (lamp>0 && ramp>0) {
					amp   = TMath::Sqrt(lamp*ramp);
					ya    = TMath::Log(la->GetAmpl()/ra->GetAmpl())*.5*att + ptBar->GetYPos();
				}

#if CUT_ON_YPOS
				//position cut by Jin Huang
				if (abs(yt)<ptBar-> GetYWidth()*2.)
					new( (*fHits)[HitNum++] ) SBSScintHit( ptBar, 0, bar, yt + ptBar->GetYPos(), tof, amp, tdiff );
				else
				{
#if DEBUG_LEVEL>=3//start show info
					Info(Here(here),"\t ypos of hit is 4 times further away from the edge of scintilator bar, ignore it. Hit info: bar=%d, y=%f, tof=%f, amp=%f, tdiff=%f",
						bar, yt + ptBar->GetYPos(), tof, amp, tdiff);
#endif//#if DEBUG_LEVEL>=3                    
				}
#else//#if CUT_ON_YPOS
				new( (*fHits)[HitNum++] ) SBSScintHit( ptBar, 0, bar, yt + ptBar->GetYPos(), tof, amp, tdiff );
#endif //#if CUT_ON_YPOS
			}
		}
	}

	return HitNum;
}

//_____________________________________________________________________________
//Int_t SBSTimingHodoscope::CombineHits( TClonesArray& tracks ) {
//	// 
//	// Put together the "complete" hits that are above fThreshold.
//	//
//	//  each combined hit has position and size (2nd moment) info
//
//	fHits->Sort();  // sorted by bar number
//
//	Int_t nhits = GetNHits();
//	if (nhits<=0) return 0;
//
//	// walk through the plane, grabbing chunks of hits above fThreshold
//	THaMultiHit *m=0;
//	Int_t nCmbHits=0;
//	int lastbar=-10;
//	for (int i=0; i<nhits; i++) {
//		SBSScintHit *h = static_cast<THaScintHit*>(fHits->At(i));
//		int bar = h->GetBarNum();
//		SBSScintBar *b = h->GetScintBar();
//
//		// test if the hit is good enough
//		if (h->GetHitEdep() < fThreshold) continue;
//		// very loose cut -- yet it is hidden but this is only for intra-plane
//		//  calculations, anyway
//		// if ( TMath::Abs( h->GetHitYPos() - b->GetYPos() ) > 2.*b->GetYWidth() ) continue;
//
//		// accepted 'good' hit
//		if (bar==lastbar+1 && m) { // neighboring bar
//			m->Add(h);
//		} else { // a new sequence
//			m = new ( (*fCombHits)[nCmbHits++] ) THaMultiHit(h);
//		}
//		lastbar = bar;
//	}
//
//	return nCmbHits;
//}

//_____________________________________________________________________________


//_____________________________________________________________________________
char* SBSTimingHodoscope::ReadNumberSignStartComment( FILE* fp, char *buf, const int len )
{
	// Read blank and comment lines 
	// return NULL if is not comment line
	int ch = fgetc(fp);  // peak ahead one character
	ungetc(ch,fp);

	if(ch=='#' || ch=='\n')
		return fgets(buf,len,fp); // read the comment
	else
		return NULL;
}


//_____________________________________________________________________________
Int_t SBSTimingHodoscope::FineMatchingHits(TClonesArray& tracks)
{
	//match hits of de and E plane with reference to tracks, modify fHits
	//process golden track if possible
	//
	//build a reference table (fTrackRef).
	//
	int n_track = tracks.GetLast()+1;   // Number of reconstructed tracks
	int matchcount=0; 

#   if FINEMATCHINGHITSTEST
	n_track=1;
#   endif

	for ( int i=0; i<n_track; i++ ) {

#       if FINEMATCHINGHITSTEST
		THaTrack* theTrack=new THaTrack(0,0,0,0);
#       else
		THaTrack* theTrack = static_cast<THaTrack*>( tracks[i] );
#       endif

		Double_t 
			pathl=kBig, 
			xc=kBig, 
			yc=kBig, 
			dx=kBig,  
			dxtmp=kBig;
		Int_t hitidx=-1;

		if ( ! CalcTrackIntercept(theTrack, pathl, xc, yc) ) { // failed to hit
			new ( (*fTrackProj)[i] )
				THaTrackProj(xc,yc,pathl,dx,hitidx);
			DEBUG_WARNING(Here("FineMatchingHits"),
				"\t Track # %d could not be projected to trigger plane",i);
			continue;
		}

		DEBUG_INFO(Here("FineMatchingHits")
			,"\t Get Track Projection Point (x,y)=(%f,%f)."
			,xc, yc);

		//look for match
		dx = kBig;
		SBSScintHit* phit;
		for ( Int_t j=0; j<GetNHits(); j++ ) {
			phit=GetHit(j);

			assert(phit);
			DEBUG_INFO(Here("FineMatchingHits")
				,"\t Trying to match with hit #%d at (x,y)=(%f,%f)."
				,j,phit->GetHitXPos(), phit->GetHitYPos());

			if(TMath::Abs(xc-phit->GetHitXPos())<fTrackAcceptanceDx and
				TMath::Abs(yc-phit->GetHitYPos())<fTrackAcceptanceDy)
			{
				//one candidate
				dxtmp=TMath::Sqrt(
					(xc-phit->GetHitXPos())*(xc-phit->GetHitXPos())+
					(yc-phit->GetHitYPos())*(yc-phit->GetHitYPos()));
				if (dx>=dxtmp)
				{//candidate for update
					dx=dxtmp;
					hitidx=j;
				}//if (dx>dxtmp)
			}
		}

		DEBUG_INFO(Here("FineMatchingHits")
			,hitidx>=0?"\t Successfully Matched with hit #%d.with minx=%f":"Matching Failed"
			,hitidx,dx);

		// record information, found or not
		new ( (*fTrackProj)[i] )
			THaTrackProj(xc,yc,pathl,dx,hitidx);
		if (hitidx>=0) matchcount++;
	}//for ( int i=0; i<n_track; i++ ) 


	//statistic
	fMatchRatioTrack=n_track==0?1:((Double_t)matchcount/(Double_t)n_track);
	DEBUG_INFO(Here("FineMatchingHits"),
		"\t %f percent of tracks have matched to hits on trigger plane.",
		fMatchRatioTrack*100);

	return kOK;
}



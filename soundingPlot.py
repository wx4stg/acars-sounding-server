#!/usr/bin/env python3
# Sam's sounding plotter
# Created 17 May 2023 by Sam Gardner <sam@wx4stg.com>


import airportsdata
import requests
from metpy.units import units
import metpy.calc as mpcalc
from metpy import plots
from metpy import constants
from metpy.io import add_station_lat_lon, parse_metar_to_dataframe
from ecape.calc import calc_ecape
from matplotlib import pyplot as plt
from matplotlib.table import table
from matplotlib.patches import Polygon
from matplotlib.patheffects import withStroke
from os import path
from pathlib import Path
import numpy as np
from datetime import datetime as dt
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeat



def makeSoundingDataset(profileData, icao=None, when=None, selectedParcel="sb"):
    # sort by decreasing pressure
    profileData = profileData.sort_values(by="LEVEL", ascending=False).reset_index(drop=True)
    if "HGHT" in profileData.keys():
        if profileData["HGHT"].is_monotonic_increasing == False:
            profileData = profileData[profileData["HGHT"] >= profileData["HGHT"].cummax()]
    if "WSPD" in profileData.keys() and "WDIR" in profileData.keys():
        profileData["u"], profileData["v"] = mpcalc.wind_components((profileData.WSPD.values * units.kt), (profileData.WDIR.values * units.deg))
    # Create xarray dataset from pandas df
    soundingDS = xr.Dataset.from_dataframe(profileData)
    if "LAT" in profileData.keys():
        # Add balloon path, if available
        soundingDS["LAT"] = soundingDS.LAT * units.degree
        soundingDS["LON"] = soundingDS.LON * units.degree
    elif icao is not None:
        # Get lat/lon from airport code, if provided
        if len(icao) == 3:
            icao = "K"+icao.upper()
        icaoDF = pd.DataFrame({"station" : icao}, index=[0])
        icaoDF = add_station_lat_lon(icaoDF)
        soundingDS.attrs["icao"] = icao
        soundingDS.attrs["LAT"] = icaoDF["latitude"].values[0] * units.degree
        soundingDS.attrs["LON"] = icaoDF["longitude"].values[0] * units.degree
    
    soundingDS["LEVEL_unitless"] = soundingDS.LEVEL
    startLevel = (soundingDS.LEVEL.data[0] // 1)
    endLevel = (soundingDS.LEVEL.data[-1] // 1)
    soundingDS = soundingDS.swap_dims({"index" : "LEVEL_unitless"})
    soundingDS = soundingDS.drop("index")
    soundingDS = soundingDS.sortby("LEVEL")
    soundingDS = soundingDS.drop_duplicates(dim="LEVEL_unitless", keep="first")
    soundingDS = soundingDS.interp(LEVEL_unitless=np.arange(endLevel, startLevel+.05, 1))
    soundingDS = soundingDS.interpolate_na(dim="LEVEL_unitless")
    soundingDS = soundingDS.sortby("LEVEL_unitless", ascending=False)
    soundingDS["index"] = np.arange(len(soundingDS.LEVEL.data))
    soundingDS = soundingDS.dropna(dim="LEVEL_unitless", how="any")
    
    # Add meteorological data
    soundingDS["LEVEL"] = soundingDS.LEVEL * units.hPa
    soundingDS["TEMP"] = soundingDS.TEMP * units.degC
    soundingDS["DWPT"] = soundingDS.DWPT * units.degC
    soundingDS["u"] = soundingDS.u * units.kt
    soundingDS["v"] = soundingDS.v * units.kt
    if "HGHT" in profileData.keys():
        soundingDS["HGHT"] = soundingDS.HGHT * units.meter
        # Calculate AGL heights if MSL heights are available, assume point 0 is the surface
        if "AGL" not in profileData.keys():
            soundingDS["AGL"] = soundingDS.HGHT - soundingDS.HGHT.data[0]
    soundingDS["WSPD"] = mpcalc.wind_speed(soundingDS.u, soundingDS.v)
    soundingDS["WDIR"] = mpcalc.wind_direction(soundingDS.u, soundingDS.v)
    soundingDS["virtT"] = mpcalc.virtual_temperature_from_dewpoint(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS["RH"] = mpcalc.relative_humidity_from_dewpoint(soundingDS.TEMP, soundingDS.DWPT)
    soundingDS["wetbulb"] = mpcalc.wet_bulb_temperature(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS["potential_temperature"] = mpcalc.potential_temperature(soundingDS.LEVEL, soundingDS.TEMP)
    soundingDS["equivalent_potential_temperature"] = mpcalc.equivalent_potential_temperature(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    # Calculate effective inflow layer
    inflowBottom = np.nan
    inflowTop = np.nan
    for i in range(len(soundingDS.LEVEL.data)):
        slicedProfile = soundingDS.isel(LEVEL_unitless=slice(i, len(soundingDS.LEVEL.data)))
        capeProfile = mpcalc.parcel_profile(slicedProfile.LEVEL, slicedProfile.TEMP[0], slicedProfile.DWPT[0]).data
        cape, cinh = mpcalc.cape_cin(slicedProfile.LEVEL.data, slicedProfile.TEMP.data, slicedProfile.DWPT.data, parcel_profile=capeProfile)
        # try:
        #     cape, cinh = mpcalc.cape_cin(slicedProfile.LEVEL.data, slicedProfile.TEMP.data, slicedProfile.DWPT.data, parcel_profile=capeProfile)
        # except ValueError as e:
        #     print(slicedProfile.LEVEL.data)
        #     print(slicedProfile.TEMP.data)
        #     print(slicedProfile.DWPT.data)
        #     testDF = pd.DataFrame({"LEVEL" : slicedProfile.LEVEL.data, "TEMP" : slicedProfile.TEMP.data, "DWPT" : slicedProfile.DWPT.data})
        #     testDF.to_csv("test.csv")
        #     mpcalc.lfc(slicedProfile.LEVEL.data, slicedProfile.TEMP.data, slicedProfile.DWPT.data, parcel_temperature_profile=capeProfile, which="bottom")
        if cape.magnitude >= 100 and cinh.magnitude >= -250:
                inflowTop = soundingDS.LEVEL.data[i]
                if np.isnan(inflowBottom):
                    inflowBottom = soundingDS.LEVEL.data[i]
        else:
            if not np.isnan(inflowBottom):
                break
    if inflowBottom == inflowTop:
        inflowBottom = np.nan
        inflowTop = np.nan
    soundingDS.attrs["inflowBottom"] = inflowBottom
    soundingDS.attrs["inflowTop"] = inflowTop

    # Calculate parcel paths, LCLs, LFCs, ELs, CAPE, CINH
    # surface-based
    sbParcelPath = mpcalc.parcel_profile(soundingDS.LEVEL, soundingDS.TEMP[0], soundingDS.DWPT[0])
    soundingDS["sbParcelPath"] = sbParcelPath
    soundingDS.attrs["sbLCL"] = mpcalc.lcl(soundingDS.LEVEL[0], soundingDS.virtT[0], soundingDS.DWPT[0])[0]
    soundingDS.attrs["sbLFC"] = mpcalc.lfc(soundingDS.LEVEL, soundingDS.virtT, soundingDS.DWPT, parcel_temperature_profile=sbParcelPath)[0]
    soundingDS.attrs["sbEL"] = mpcalc.el(soundingDS.LEVEL, soundingDS.virtT, soundingDS.DWPT, parcel_temperature_profile=sbParcelPath)[0]
    soundingDS.attrs["sbCAPE"], soundingDS.attrs["sbCINH"] = mpcalc.surface_based_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)

    # most unstable
    initPressure, initTemp, initDewpoint, initIdx = mpcalc.most_unstable_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["mu_initPressure"], soundingDS.attrs["mu_initTemp"], soundingDS.attrs["mu_initDewpoint"] = initPressure, initTemp, initDewpoint
    initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
    muParcelPath = xr.full_like(soundingDS.TEMP, np.nan * units.K)
    muParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], initTemp, initDewpoint)
    soundingDS["muParcelPath"] = muParcelPath
    soundingDS.attrs["muLCL"] = mpcalc.lcl(initPressure, initTemp, initDewpoint)[0]
    soundingDS.attrs["muLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=muParcelPath[initIdx:])[0]
    soundingDS.attrs["muEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=muParcelPath[initIdx:])[0]
    soundingDS.attrs["muCAPE"], soundingDS.attrs["muCINH"] = mpcalc.most_unstable_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    
    # 100-hPa mixed layer
    mlParcelPath = xr.full_like(soundingDS.TEMP, np.nan * units.K)
    if np.nanmax(soundingDS.LEVEL.data) - np.nanmin(soundingDS.LEVEL.data) > 100 * units.hPa:
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
        initIdx = len(soundingDS.where(soundingDS.LEVEL > initPressure, drop=True).LEVEL.data)
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
        mlParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], initVirtT, initDewpoint)
        soundingDS.attrs["mlLCL"] = mpcalc.lcl(initPressure, initVirtT, initDewpoint)[0]
        soundingDS.attrs["mlLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=mlParcelPath[initIdx:])[0]
        soundingDS.attrs["mlEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=mlParcelPath[initIdx:])[0]
        soundingDS.attrs["mlCAPE"], soundingDS.attrs["mlCINH"] = mpcalc.mixed_layer_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    else:
        soundingDS.attrs["mlLCL"] = np.nan * units.hPa
        soundingDS.attrs["mlLFC"] = np.nan * units.hPa
        soundingDS.attrs["mlEL"] = np.nan * units.hPa
        soundingDS.attrs["mlCAPE"] = np.nan * units.joule/units.kilogram
        soundingDS.attrs["mlCINH"] = np.nan * units.joule/units.kilogram
    soundingDS["mlParcelPath"] = mlParcelPath

    # effective inflow layer
    inflowParcelPath = xr.full_like(soundingDS.TEMP, np.nan * units.K)
    if not np.isnan(inflowBottom):
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_start_pressure=inflowBottom, depth=(inflowBottom - inflowTop))
        initIdx = len(soundingDS.where(soundingDS.LEVEL > initPressure, drop=True).LEVEL.data)
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
        inflowParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx], soundingDS.DWPT[initIdx])
        soundingDS.attrs["inLCL"] = mpcalc.lcl(initPressure, initVirtT, initDewpoint)[0]
        soundingDS.attrs["inLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=inflowParcelPath[initIdx:])[0]
        soundingDS.attrs["inEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=inflowParcelPath[initIdx:])[0]
        soundingDS.attrs["inCAPE"], soundingDS.attrs["inCINH"] = mpcalc.mixed_layer_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_start_pressure=inflowBottom, depth=(inflowBottom - inflowTop))
    else:
        inflowParcelPath = inflowParcelPath * units.degK
        soundingDS.attrs["inLCL"] = np.nan * units.hPa
        soundingDS.attrs["inLFC"] = np.nan * units.hPa
        soundingDS.attrs["inEL"] = np.nan * units.hPa
        soundingDS.attrs["inCAPE"], soundingDS.attrs["inCINH"] = np.nan * units.joule/units.kilogram, np.nan * units.joule/units.kilogram
    soundingDS["inParcelPath"] = inflowParcelPath
    # Cloud Layer heights
    soundingDS.attrs["cloudLayerBottom"] = soundingDS.attrs[selectedParcel+"LCL"]
    soundingDS.attrs["cloudLayerTop"] = soundingDS.attrs[selectedParcel+"EL"]

    # pwat
    soundingDS.attrs["pwat"] = mpcalc.precipitable_water(soundingDS.LEVEL, soundingDS.DWPT).to(units.inch)
    # storm motion
    if np.nanmax(soundingDS.AGL.data) > 6000 * units.meter:
        soundingDS.attrs["bunkers_RM"], soundingDS.attrs["bunkers_LM"], soundingDS.attrs["zeroToSixMean"] = mpcalc.bunkers_storm_motion(soundingDS.LEVEL, soundingDS.u, soundingDS.v, soundingDS.HGHT)
        soundingDS.attrs["corfidi_up"], soundingDS.attrs["corfidi_down"] = mpcalc.corfidi_storm_motion(soundingDS.LEVEL, soundingDS.u, soundingDS.v, soundingDS.HGHT)
        # SRH
        soundingDS.attrs["RM_SRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=3000*units.meter, storm_u=soundingDS.bunkers_RM[0], storm_v=soundingDS.bunkers_RM[1])[2]
        soundingDS.attrs["MW_SRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=3000*units.meter, storm_u=soundingDS.zeroToSixMean[0], storm_v=soundingDS.zeroToSixMean[1])[2]
        soundingDS.attrs["LM_SRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=3000*units.meter, storm_u=soundingDS.bunkers_LM[0], storm_v=soundingDS.bunkers_LM[1])[2]
        # RH
        soundingDS.attrs["LL_RH"] = soundingDS.where(soundingDS.LEVEL >= soundingDS.LEVEL[0] - 100*units.hPa, drop=True).RH.data.mean()
        soundingDS.attrs["ML_RH"] = soundingDS.where(soundingDS.LEVEL >= soundingDS.LEVEL[0] - 350*units.hPa, drop=True).RH.data.mean()
        # which bunkers is favored
        if soundingDS.attrs["RM_SRH"] >= soundingDS.attrs["LM_SRH"]:
            soundingDS.attrs["favored_motion"] = "RM"
        else:
            soundingDS.attrs["favored_motion"] = "LM"
    else:
        soundingDS.attrs["bunkers_RM"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["bunkers_LM"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["zeroToSixMean"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["corfidi_up"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["corfidi_down"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["RM_SRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["MW_SRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["LM_SRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["LL_RH"] = np.nan * units.percent
        soundingDS.attrs["ML_RH"] = np.nan * units.percent
        soundingDS.attrs["favored_motion"] = None
    
    if np.nanmax(soundingDS.AGL.data) > 1000 * units.meter:
        soundingDS.attrs["sfc_to_one_shear"] = mpcalc.bulk_shear(soundingDS.LEVEL, soundingDS.u, soundingDS.v, depth=1000 * units.meter)
    else:
        soundingDS.attrs["sfc_to_one_shear"] = [np.nan * units.knot, np.nan * units.knot]
    if np.nanmax(soundingDS.AGL.data) > 6000 * units.meter:
        soundingDS.attrs["sfc_to_six_shear"] = mpcalc.bulk_shear(soundingDS.LEVEL, soundingDS.u, soundingDS.v, depth=6000 * units.meter)
    else:
        soundingDS.attrs["sfc_to_six_shear"] = [np.nan * units.knot, np.nan * units.knot]
    if np.nanmax(soundingDS.AGL.data) > 8000 * units.meter:
        soundingDS.attrs["sfc_to_eight_shear"] = mpcalc.bulk_shear(soundingDS.LEVEL, soundingDS.u, soundingDS.v, depth=8000 * units.meter)
    else:
        soundingDS.attrs["sfc_to_eight_shear"] = [np.nan * units.knot, np.nan * units.knot]

    # AGL versions of the pressure levels
    for key, value in soundingDS.attrs.copy().items():
        if "LCL" in key or "LFC" in key or "EL" in key or "inflow" in key or "cloudLayer" in key:
            if np.isnan(value):
                soundingDS.attrs[key+"_agl"] = np.nan * units.meter
            else:
                soundingDS.attrs[key+"_agl"] = soundingDS.interp(LEVEL_unitless=value.to(units.hPa).magnitude).AGL.data * units.meter

    # Other assorted params needed for SHARPpy's hazard type decision tree
    soundingDS.attrs["sfc_to_one_LR"] = -((soundingDS.TEMP.data[0] - soundingDS.where(soundingDS.AGL <= 1000 * units.meter, drop=True).TEMP.data[-1])/(soundingDS.AGL.data[0] - soundingDS.where(soundingDS.AGL <= 1000 * units.meter, drop=True).AGL.data[-1]).to(units.km))
    soundingDS.attrs["five_to_seven_LR"] = -((soundingDS.where(soundingDS.LEVEL >= 500 * units.hPa, drop=True).TEMP.data[-1] - soundingDS.where(soundingDS.LEVEL >= 700 * units.hPa, drop=True).TEMP.data[-1])/(soundingDS.where(soundingDS.LEVEL >= 500 * units.hPa, drop=True).HGHT.data[-1] - soundingDS.where(soundingDS.LEVEL >= 700 * units.hPa, drop=True).HGHT.data[-1]).to(units.km))
    if np.min(soundingDS.TEMP.data) < 0 * units.degC:
        soundingDS.attrs["freezing_level_agl"] = soundingDS.where(soundingDS.TEMP <= 0 * units.degC, drop=True).AGL.data[0]
    else: 
        soundingDS.attrs["freezing_level_agl"] = np.nan * units.meter
    if soundingDS.favored_motion is not None:
        soundingDS.attrs["favored1kmSRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=1000*units.meter, storm_u=soundingDS.attrs["bunkers_"+soundingDS.favored_motion][0], storm_v=soundingDS.attrs["bunkers_"+soundingDS.favored_motion][1])[2]
        soundingDS.attrs["fixed_stp"] = mpcalc.significant_tornado(soundingDS.sbCAPE, soundingDS.sbLCL_agl, soundingDS.favored1kmSRH, mpcalc.wind_speed(*soundingDS.sfc_to_six_shear)).to_base_units().magnitude[0]
        cinTerm = 1
        if soundingDS.mlCINH > -50 * units.joule/units.kilogram:
            cinTerm = 1
        elif soundingDS.mlCINH < -200 * units.joule/units.kilogram:
            cinTerm = 0
        else:
            cinTerm = ((soundingDS.mlCINH.magnitude + 200.) / 150.)
        soundingDS.attrs["effective_stp"] = (soundingDS.fixed_stp * cinTerm)
    else:
        soundingDS.attrs["favored1kmSRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["fixed_stp"] = 0 * units.dimensionless
        soundingDS.attrs["effective_stp"] = 0 * units.dimensionless
    if not np.isnan(soundingDS.attrs["sfc_to_six_shear"][0]):
        shearMag = mpcalc.wind_speed(*soundingDS.attrs["sfc_to_six_shear"])
    layerForCalc = soundingDS.where((soundingDS.LEVEL <= soundingDS.inflowBottom) & (soundingDS.LEVEL >= soundingDS.inflowTop), drop=True)
    if len(layerForCalc.LEVEL.data) > 0 and soundingDS.attrs["favored_motion"] is not None:
        bottom = layerForCalc.AGL.data[0]
        top = layerForCalc.AGL.data[-1]
        depth = (top - bottom)
        favoredEILSRH = mpcalc.storm_relative_helicity(layerForCalc.AGL, layerForCalc.u, layerForCalc.v, bottom=bottom, depth=depth, storm_u=soundingDS.attrs["bunkers_"+soundingDS.attrs["favored_motion"]][0], storm_v=soundingDS.attrs["bunkers_"+soundingDS.attrs["favored_motion"]][1])[2]
        soundingDS.attrs["scp"] = mpcalc.supercell_composite(soundingDS.muCAPE, favoredEILSRH, shearMag)[0]
    else:
        soundingDS.attrs["scp"] = 0 * units.dimensionless
    soundingDS.attrs["k"] = mpcalc.k_index(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["totaltotals"] = mpcalc.total_totals_index(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["sweat"] = mpcalc.sweat_index(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, soundingDS.WSPD, soundingDS.WDIR)[0]
    try:
        _, _, soundingDS.attrs["convT"] = mpcalc.ccl(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    except IndexError as e:
        print(e)
        soundingDS.attrs["convT"] = np.nan * units.degC
    # DCAPE
    dcape_profile = xr.full_like(soundingDS.TEMP, np.nan)
    if np.nanmax(soundingDS.LEVEL.data) > 700 * units.hPa and np.nanmin(soundingDS.LEVEL.data) < 500 * units.hPa:
        dcape_result  = mpcalc.down_cape(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
        dcape_quantity = dcape_result[0]
        dcape_profile[:len(dcape_result[2])] = dcape_result[2]
    else:
        dcape_quantity = np.nan * units.joule/units.kilogram
    soundingDS.attrs["dcape"] = dcape_quantity
    soundingDS["dcape_profile"] = dcape_profile
    soundingDS = soundingDS.drop('index')
    return soundingDS
    

def plotParams(profileData, ax):
    ax.text(0.01, 0.79, f"SWEAT:\n{profileData.sweat.magnitude:.1f}", ha="left", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.4, 0.79, f"SCP:\n{profileData.scp.magnitude:.1f}", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.65, 0.79, f"K:\n{profileData.k.magnitude:.1f}", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.99, 0.79, f"eSTP:\n{profileData.effective_stp:.1f}", ha="right", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    if not np.isnan(profileData.dcape_profile[0]):
        downMax = (2*profileData.dcape.to('J/kg').magnitude)**0.5
        ax.text(0.5, 0.6, f"DCAPE:\n{int(profileData.dcape.to('J/kg').magnitude)} J/kg, {downMax:.1f} m/s, {int(profileData.dcape_profile[0].to(units.degF).magnitude)}F", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    else:
        ax.text(0.5, 0.6, f"DCAPE:\nN/A", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0, 0.44, f"PWAT:\n{profileData.pwat.to('in').magnitude:.2f} in", ha="left", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.5, 0.44, f"TT:\n{profileData.totaltotals.magnitude:.1f}", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    if not np.isnan(profileData.convT):
        ax.text(0.99, 0.44, f"ConvT:\n{int(profileData.convT.to(units.degF).magnitude)}°F", ha="right", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    else:
        ax.text(0.99, 0.44, f"ConvT:\nN/A", ha="right", va="top", transform=ax.transAxes, clip_on=False, zorder=5)

    TorLR = -((profileData.TEMP.data[0] - profileData.where(profileData.AGL <= 500 * units.meter, drop=True).TEMP.data[-1])/(profileData.AGL.data[0] - profileData.where(profileData.AGL <= 500 * units.meter, drop=True).AGL.data[-1]).to(units.km))
    LLLR = -((profileData.TEMP.data[0] - profileData.where(profileData.AGL <= 3000 * units.meter, drop=True).TEMP.data[-1])/(profileData.AGL.data[0] - profileData.where(profileData.AGL <= 3000 * units.meter, drop=True).AGL.data[-1]).to(units.km))
    MLLR = -((profileData.where(profileData.AGL <= 3000 * units.meter, drop=True).TEMP.data[-1] - profileData.where(profileData.AGL <= 6000 * units.meter, drop=True).TEMP.data[-1])/(profileData.where(profileData.AGL <= 3000 * units.meter, drop=True).AGL.data[-1] - profileData.where(profileData.AGL <= 6000 * units.meter, drop=True).AGL.data[-1]).to(units.km))
    ax.text(0.5, 0.2, "Lapse Rates (°C/km):", ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0, 0.01, f"0->.5km\n{TorLR.magnitude:.1f}", ha="left", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.5, 0.01, f"0->3km\n{LLLR.magnitude:.1f}", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.99, 0.01, f"3->6km\n{MLLR.magnitude:.1f}", ha="right", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)


def plotThermoynamics(profileData, ax, parcelType="sb"):
    thermodynamicsTableContent = [["Parcel", "CAPE", "CINH", "ECAPE", "LCL", "LFC", "EL", "0->3km\nCAPE"]]
    variables = [
        ("Surface Based", mpcalc.surface_based_cape_cin),
    ]
    if not np.isnan(profileData.mlCAPE):
        variables.append(("Mixed Layer", mpcalc.mixed_layer_cape_cin))
    else:
        variables.append(("Mixed Layer", None))
    if not np.isnan(profileData.muCAPE):
        variables.append(("Most Unstable", mpcalc.most_unstable_cape_cin))
    else:
        variables.append(("Most Unstable", None))
    for var in variables:
        caption, cape_func = var
        if cape_func is None:
            thermodynamicsTableContent.append([caption, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
        capeType = "".join([letter.lower() for letter in caption if letter.isupper()])
        try:
            ecape = calc_ecape(profileData.HGHT.data, profileData.LEVEL.data, profileData.TEMP.data, mpcalc.specific_humidity_from_dewpoint(profileData.LEVEL.data, profileData.DWPT.data), profileData.u.data, profileData.v.data, cape_type=caption.lower().replace(" ", "_"))
        except Exception as e:
            ecape = 0 * units.joule/units.kilogram
            print(e)
        lvlBelow3 = profileData.where(profileData.AGL <= 3000*units.meter, drop=True).LEVEL.data
        tempBelow3 = profileData.where(profileData.AGL <= 3000*units.meter, drop=True).TEMP.data
        dwptBelow3 = profileData.where(profileData.AGL <= 3000*units.meter, drop=True).DWPT.data
        try:
            cape3km = cape_func(lvlBelow3, tempBelow3, dwptBelow3)[0]
        except Exception as e:
            cape3km = np.nan * units.joule/units.kilogram
            

        if not np.isnan(profileData.attrs[capeType+'CAPE']):
            wmax = (2*profileData.attrs[capeType+'CAPE'].to('J/kg').magnitude)**0.5
            capeLbl = f"{int(profileData.attrs[capeType+'CAPE'].to('J/kg').magnitude)} J/kg\n{wmax:.1f} m/s"
        else:
            capeLbl = "0 J/kg\n0 m/s"
        if not np.isnan(profileData.attrs[capeType+'CINH']):
            cinhLbl = f"{int(profileData.attrs[capeType+'CINH'].to('J/kg').magnitude)} J/kg"
        else:
            cinhLbl = "0 J/kg"
        if not np.isnan(ecape):
            wmax = (2*ecape.to('J/kg').magnitude)**0.5
            ecapeLbl = f"{int(ecape.to('J/kg').magnitude)} J/kg\n{wmax:.1f} m/s"
        else:
            ecapeLbl = "0 J/kg\n0 m/s"
        if not np.isnan(profileData.attrs[capeType+'LCL_agl']):
            lclLbl = f"{int(profileData.attrs[capeType+'LCL_agl'].to(units.meter).magnitude)} m\n{int(profileData.attrs[capeType+'LCL'].to(units.hPa).magnitude)} hPa"
        else:
            lclLbl = "N/A"
        if not np.isnan(profileData.attrs[capeType+'LFC_agl']):
            lfcLbl = f"{int(profileData.attrs[capeType+'LFC_agl'].to(units.meter).magnitude)} m\n{int(profileData.attrs[capeType+'LFC'].to(units.hPa).magnitude)} hPa"
        else:
            lfcLbl = "N/A"
        if not np.isnan(profileData.attrs[capeType+'EL_agl']):
            elLbl = f"{int(profileData.attrs[capeType+'EL_agl'].to(units.meter).magnitude)} m\n{int(profileData.attrs[capeType+'EL'].to(units.hPa).magnitude)} hPa"
        else:
            elLbl = "N/A"
        if not np.isnan(cape3km):
            cape3kmLbl = f"{int(cape3km.magnitude)} J/kg"
        else:
            cape3kmLbl = "0 J/kg"

        thermodynamicsTableContent.append([caption, capeLbl, cinhLbl, ecapeLbl, lclLbl, lfcLbl, elLbl, cape3kmLbl])


    if not np.isnan(profileData.inflowBottom):
        caption = "Effective Inflow"
        cape_func = mpcalc.mixed_layer_cape_cin
        try:
            ecape = calc_ecape(profileData.HGHT.data, profileData.LEVEL.data, profileData.TEMP.data, mpcalc.specific_humidity_from_dewpoint(profileData.LEVEL.data, profileData.DWPT.data), profileData.u.data, profileData.v.data, undiluted_cape=profileData.mlCAPE)
        except:
            ecape = 0 * units.joule/units.kilogram
        cape3km = cape_func(profileData.where(profileData.AGL <= 3000*units.meter, drop=True).LEVEL, profileData.where(profileData.AGL <= 3000*units.meter, drop=True).TEMP, profileData.where(profileData.AGL <= 3000*units.meter, drop=True).DWPT, bottom=profileData.inflowBottom, depth=(profileData.inflowBottom - profileData.inflowTop))[0]
        capeType = "in"
        if not np.isnan(profileData.attrs[capeType+'CAPE']):
            vmax = (2*profileData.attrs[capeType+'CAPE'].to('J/kg').magnitude)**0.5
            capeLbl = f"{int(profileData.attrs[capeType+'CAPE'].to('J/kg').magnitude)} J/kg\n{vmax:.1f} m/s"
        else:
            capeLbl = "0 J/kg\n0 m/s"
        if not np.isnan(profileData.attrs[capeType+'CINH']):
            cinhLbl = f"{int(profileData.attrs[capeType+'CINH'].to('J/kg').magnitude)} J/kg"
        else:
            cinhLbl = "0 J/kg"
        if not np.isnan(ecape):
            wmax = (2*ecape.to('J/kg').magnitude)**0.5
            ecapeLbl = f"{int(ecape.to('J/kg').magnitude)} J/kg\n{wmax:.1f} m/s"
        else:
            ecapeLbl = "0 J/kg\n0 m/s"
        if not np.isnan(profileData.attrs[capeType+'LCL_agl']):
            lclLbl = f"{int(profileData.attrs[capeType+'LCL_agl'].to(units.meter).magnitude)} m\n{int(profileData.attrs[capeType+'LCL'].to(units.hPa).magnitude)} hPa"
        else:
            lclLbl = "N/A"
        if not np.isnan(profileData.attrs[capeType+'LFC_agl']):
            lfcLbl = f"{int(profileData.attrs[capeType+'LFC_agl'].to(units.meter).magnitude)} m\n{int(profileData.attrs[capeType+'LFC'].to(units.hPa).magnitude)} hPa"
        else:
            lfcLbl = "N/A"
        if not np.isnan(profileData.attrs[capeType+'EL_agl']):
            elLbl = f"{int(profileData.attrs[capeType+'EL_agl'].to(units.meter).magnitude)} m\n{int(profileData.attrs[capeType+'EL'].to(units.hPa).magnitude)} hPa"
        else:
            elLbl = "N/A"
        if not np.isnan(cape3km):
            cape3kmLbl = f"{int(cape3km.magnitude)} J/kg"
        else:
            cape3kmLbl = "0 J/kg"
        thermodynamicsTableContent.append([caption, capeLbl, cinhLbl, ecapeLbl, lclLbl, lfcLbl, elLbl, cape3kmLbl])
    else:
        thermodynamicsTableContent.append(["Effective Inflow", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
    
    thermodynamicsTable = table(ax, bbox=[0, 0, 1, 0.89], cellText=thermodynamicsTableContent, cellLoc="center")
    thermodynamicsTable.auto_set_font_size(False)


def plotDynamics(profileData, ax):
    dynamicsTableContent = [["Bunkers Right//0-6km Mean Wind//Bunkers Left", "Bulk\nWind\nDifference", "Mean Wind", "Storm\nRelative\nHelicity", "Storm\nRelative\nWind", "Horizontal\nVorticity\n(Streamwise%)"]]
    levelsToSample = [(profileData.inflowBottom_agl, profileData.inflowTop_agl), (0*units.meter, 500*units.meter), (0*units.meter, 1000*units.meter), (0*units.meter, 3000*units.meter), (0*units.meter, 6000*units.meter), (0*units.meter, 9000*units.meter), (profileData.cloudLayerBottom_agl, profileData.cloudLayerTop_agl)]
    rm = profileData.attrs["bunkers_RM"]
    mw = profileData.attrs["zeroToSixMean"]
    lm = profileData.attrs["bunkers_LM"]
    for i in range(len(levelsToSample)):
        thisLayerRow = []
        bottom, top = levelsToSample[i]
        depth = top - bottom
        if i == 0:
            thisLayerRow.append("Effective\nInflow Layer")
        elif i == len(levelsToSample)-1:
            thisLayerRow.append("Cloud Layer\n(LCL->EL)")
        elif i == 1:
            thisLayerRow.append(f"SFC->500m")
        else:
            thisLayerRow.append(f"SFC->{int(top.to(units.km).magnitude)}km")
        layerForCalc = profileData.where((profileData.AGL >= bottom) & (profileData.AGL <= top), drop=True)
        layerForCalc = layerForCalc.where((np.isnan(layerForCalc.LEVEL) == False) & (np.isnan(layerForCalc.u) == False) & (np.isnan(layerForCalc.v) == False), drop=True)
        if len(layerForCalc.LEVEL.data) <= 2:
            thisLayerRow.extend(["N/A", "N/A", "N/A", "N/A", "N/A"])
            dynamicsTableContent.append(thisLayerRow)
            continue
        bottomp, topp = layerForCalc.LEVEL[0], layerForCalc.LEVEL[-1]
        depthp = bottomp - topp
        shearU, shearV = mpcalc.bulk_shear(layerForCalc.LEVEL, layerForCalc.u, layerForCalc.v, height=layerForCalc.AGL, bottom=bottomp, depth=depthp)
        shearMag, shearDir = mpcalc.wind_speed(shearU, shearV), mpcalc.wind_direction(shearU, shearV)
        thisLayerRow.append(f"{shearMag.magnitude:.1f}kt//{shearDir.magnitude:.1f}°")
        meanWindu, meanWindv = mpcalc.mean_pressure_weighted(layerForCalc.LEVEL, layerForCalc.u, layerForCalc.v, height=layerForCalc.AGL, bottom=bottomp, depth=depthp)
        meanWindMag, meanWindDir = mpcalc.wind_speed(meanWindu, meanWindv), mpcalc.wind_direction(meanWindu, meanWindv)
        thisLayerRow.append(f"{meanWindMag.magnitude:.1f}kt//{meanWindDir.magnitude:.1f}°")
        RMstormRelativeHelicity = mpcalc.storm_relative_helicity(layerForCalc.AGL, layerForCalc.u, layerForCalc.v, bottom=bottom, depth=depth, storm_u=layerForCalc.bunkers_RM[0], storm_v=layerForCalc.bunkers_RM[1])[2]
        MWstormRelativeHelicity = mpcalc.storm_relative_helicity(layerForCalc.AGL, layerForCalc.u, layerForCalc.v, bottom=bottom, depth=depth, storm_u=layerForCalc.zeroToSixMean[0], storm_v=layerForCalc.zeroToSixMean[1])[2]
        LMstormRelativeHelicity = mpcalc.storm_relative_helicity(layerForCalc.AGL, layerForCalc.u, layerForCalc.v, bottom=bottom, depth=depth, storm_u=layerForCalc.bunkers_LM[0], storm_v=layerForCalc.bunkers_LM[1])[2]
        thisLayerRow.append(str(int(RMstormRelativeHelicity.magnitude))+"$m^2 s^{-2}$//"+str(int(MWstormRelativeHelicity.magnitude))+"$m^2 s^{-2}$//"+str(int(LMstormRelativeHelicity.magnitude))+"$m^2 s^{-2}$")
        if not np.isnan(rm[0]):
            RMsrwU, RMsrwV = (meanWindu-rm[0]), (meanWindv-rm[1])
            RMsrwMag, RMsrwDir = mpcalc.wind_speed(RMsrwU, RMsrwV).to(units.kt), mpcalc.wind_direction(RMsrwU, RMsrwV)
            MWsrwU, MWsrwV = (meanWindu-mw[0]), (meanWindu-mw[1])
            MWsrwMag, MWsrwDir = mpcalc.wind_speed(MWsrwU, MWsrwV).to(units.kt), mpcalc.wind_direction(MWsrwU, MWsrwV)
            LMsrwU, LMsrwV = (meanWindu-lm[0]), (meanWindv-lm[1])
            LMsrwMag, LMsrwDir = mpcalc.wind_speed(LMsrwU, LMsrwV).to(units.kt), mpcalc.wind_direction(LMsrwU, LMsrwV)
            thisLayerRow.append(f"{int(RMsrwMag.magnitude)}kt/{int(RMsrwDir.magnitude)}°//{int(MWsrwMag.magnitude)}kt/{int(MWsrwDir.magnitude)}°//{int(LMsrwMag.magnitude)}kt/{int(LMsrwDir.magnitude)}°")
        else:
            thisLayerRow.append("N/A")
        dvdz = mpcalc.first_derivative(layerForCalc.v.data.to(units.meter/units.sec), x=(layerForCalc["AGL"].values*units.meter))
        dvdz = mpcalc.mean_pressure_weighted(layerForCalc.LEVEL, dvdz, height=(layerForCalc["AGL"].values*units.meter), bottom=bottomp, depth=depthp)[0]
        dudz = mpcalc.first_derivative(layerForCalc.u.data.to(units.meter/units.sec), x=(layerForCalc["AGL"].values*units.meter))
        dudz = mpcalc.mean_pressure_weighted(layerForCalc.LEVEL, dudz, height=(layerForCalc["AGL"].values*units.meter), bottom=bottomp, depth=depthp)[0]
        
        horizVort = ((-dvdz)**2 + (dudz)**2)**0.5
        
        if not np.isnan(rm[0]):
            rmstreamwiseVort = ((-dvdz * RMsrwU) + (dudz * RMsrwV))/((RMsrwU**2 + RMsrwV**2)**(0.5))
            rmcrosswiseVort = (horizVort**2 - rmstreamwiseVort**2)**0.5
            rmstreamwisePercent = (rmstreamwiseVort/(rmstreamwiseVort+rmcrosswiseVort))*100
            
            mwstreamwiseVort = ((-dvdz * MWsrwU) + (dudz * MWsrwV))/((MWsrwU**2 + MWsrwV**2)**(0.5))
            mwcrosswiseVort = (horizVort**2 - mwstreamwiseVort**2)**0.5
            mwstreamwisePercent = (mwstreamwiseVort/(mwstreamwiseVort+mwcrosswiseVort))*100

            lmstreamwiseVort = ((-dvdz * LMsrwU) + (dudz * LMsrwV))/((LMsrwU**2 + LMsrwV**2)**(0.5))
            lmcrosswiseVort = (horizVort**2 - lmstreamwiseVort**2)**0.5
            lmstreamwisePercent = np.abs(lmstreamwiseVort/(lmstreamwiseVort+lmcrosswiseVort))*100

            thisLayerRow.append(str(round(horizVort.magnitude, 3))+"$s^{-1}$//"+str(int(rmstreamwisePercent.magnitude))+"%//"+str(int(mwstreamwisePercent.magnitude))+"%//"+str(int(lmstreamwisePercent.magnitude))+"%")
        else:
            thisLayerRow.append(str(round(horizVort.magnitude, 3))+"$s^{-1}$//N/A//N/A//N/A")
        dynamicsTableContent.append(thisLayerRow)
    dynamicsTable = table(ax, bbox=[0, 0, 1, 1], cellText=np.empty((8,6), dtype=str), cellLoc="center")
    if profileData.favored_motion == "RM":
        rmcolor = "red"
        lmcolor = "gray"
    else:
        rmcolor = "gray"
        lmcolor = "blue"
    for rowNum in range(len(dynamicsTableContent)):
        for colNum in range(len(dynamicsTableContent[rowNum])):
            if rowNum == 0:
                if colNum == 0:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    rm = textArr[0]
                    mw = textArr[1]
                    lm = textArr[2]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.3+rowNum)/len(dynamicsTableContent)), "Color Key", color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.4+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="center", va="center", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.6+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="center", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.7+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="center", va="top", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            elif colNum == 0:
                if rowNum == 1:
                    color = "teal"
                elif rowNum == 2:
                    color = "fuchsia"
                elif rowNum == 3:
                    color = "firebrick"
                elif rowNum == 4:
                    color = "limegreen"
                elif rowNum == 5:
                    color = "goldenrod"
                elif rowNum == 6:
                    color = "darkturquoise"
                elif rowNum == 7:
                    color = "mediumpurple"
                else:
                    color = "black"
                ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], color=color, ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            elif colNum == 1:
                if "//" in dynamicsTableContent[rowNum][colNum]:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    mag = textArr[0]
                    dir = textArr[1]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mag, color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dir, color="gray", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            elif colNum == 2:
                if "//" in dynamicsTableContent[rowNum][colNum]:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    mag = textArr[0]
                    dir = textArr[1]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mag, color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dir, color="black", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            elif colNum == 3:
                if "//" in dynamicsTableContent[rowNum][colNum]:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    rm = textArr[0]
                    mw = textArr[1]
                    lm = textArr[2]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.4+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.6+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            elif colNum == 4:
                if "//" in dynamicsTableContent[rowNum][colNum]:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    rm = textArr[0]
                    mw = textArr[1]
                    lm = textArr[2]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.4+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.6+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            elif colNum == 5:
                if "//" in dynamicsTableContent[rowNum][colNum]:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    hv = textArr[0]
                    rm = textArr[1]
                    mw = textArr[2]
                    lm = textArr[3]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), hv, color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.3+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="right", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.7+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="left", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
            else:
                ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)

def plotPartialThickness(profileData, ax):
    lowLevelBottom = profileData.interp(LEVEL_unitless=1000).AGL.data * units.meter
    boundaryLevel = profileData.interp(LEVEL_unitless=850).AGL.data * units.meter
    midLevelTop = profileData.interp(LEVEL_unitless=700).AGL.data * units.meter
    if np.isnan(lowLevelBottom) or np.isnan(boundaryLevel) or np.isnan(midLevelTop):
        midLevelThickness = np.nan
        lowLevelThickness = np.nan
    else:
        midLevelThickness = (midLevelTop - boundaryLevel).to(units.meter).magnitude
        lowLevelThickness = (boundaryLevel - lowLevelBottom).to(units.meter).magnitude
        if midLevelThickness >=1560:
            midLevelThickness = 1560
        if midLevelThickness <= 1525:
            midLevelThickness = 1525
        if lowLevelThickness >= 1315:
            lowLevelThickness = 1315
        if lowLevelThickness <= 1281:
            lowLevelThickness = 1281
    ax.scatter(midLevelThickness, lowLevelThickness, color="gold", edgecolor="black", marker="*", s=125, zorder=10)
    ax.text(0.5, 0, "850-700 hPa Thickness", color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=6)
    ax.text(0.01, 0.5, "1000-850 hPa Thickness", color="black", ha="left", va="center", transform=ax.transAxes, clip_on=False, zorder=5, rotation=90, fontsize=6)
    snowPoly = Polygon(np.array([
        [0,0],
        [0, 1537.375],
        [1500, 1303],
        [1532, 1298],
        [1537, 1290],
        [1542, 1278],
        [1545, 1260],
        [1755, 0]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(snowPoly)
    ax.text(1527, 1287, "Snow", color="black", ha="left", va="bottom", fontsize=8, zorder=2)
    unknownPoly = Polygon(np.array([
        [0, 20000],
        [0, 1537.375],
        [1500, 1303],
        [1532, 1298],
        [1530, 1312],
        [1538, 1312],
        [1538, 20000]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(unknownPoly)
    ax.text(1525, 1314, "Unknown", color="black", ha="left", va="top", fontsize=8, zorder=2)
    snowRainPoly = Polygon(np.array([
        [1530, 1312],
        [1538, 1312],
        [1543, 1290],
        [1537, 1290],
        [1532, 1298]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(snowRainPoly)
    ax.text(1536, 1300, "Snow +\nRain", color="black", ha="center", va="center", fontsize=8, zorder=2)
    wintryMixPoly = Polygon(np.array([
        [1538, 1312],
        [1550, 1310],
        [1550, 1290],
        [1543, 1290]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(wintryMixPoly)
    ax.text(1545, 1300, "Wintry\nMix", color="black", ha="center", va="center", fontsize=8, zorder=2)
    rainPoly = Polygon(np.array([
        [1538, 1312],
        [1550, 1310],
        [1580, 1314],
        [20000, 3770],
        [20000, 20000],
        [1538, 20000]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(rainPoly)
    ax.text(1554, 1314, "Rain", color="black", ha="right", va="top", fontsize=8, zorder=2)
    snowSleetPoly = Polygon(np.array([
        [1537, 1290],
        [1550, 1290],
        [1550, 1282],
        [1562, 1260],
        [(24742/11), 0],
        [1755, 0],
        [1545, 1260],
        [1542, 1278]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(snowSleetPoly)
    ax.text(1544, 1285, "Snow+\nSleet", color="black", ha="center", va="center", fontsize=8, zorder=2)
    sleetFZRAPoly = Polygon(np.array([
        [1550, 1293],
        [1580, 1290],
        [14480, 0],
        [(24742/11), 0],
        [1562, 1260],
        [1550, 1282],
        [1550, 1290]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(sleetFZRAPoly)
    ax.text(1555, 1285, "Sleet+\nFrz. Rain", color="black", ha="center", va="center", fontsize=8, zorder=2)
    FZRASleetPoly = Polygon(np.array([
        [1550, 1293],
        [1580, 1290],
        [1850, 1263],
        [1580, 1299],
        [1550, 1303]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(FZRASleetPoly)
    ax.text(1555, 1295, "Frz. Rain+\nSleet", color="black", ha="center", va="center", fontsize=8, zorder=2)
    FRZARainPoly = Polygon(np.array([
        [1550, 1310],
        [1580, 1314],
        [20000, 3770],
        [20000, 0],
        [14480, 0],
        [1850, 1263],
        [1580, 1299],
        [1550, 1303]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(FRZARainPoly)
    ax.text(1555, 1305, "Frz. Rain", color="black", ha="center", va="center", fontsize=8, zorder=2)
    ax.set_xlim([1525, 1560])
    ax.set_ylim([1281, 1315])
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, zorder=0)
    isSaturatedAnywhere = False
    if "RH" in profileData.keys():
        for top in (np.arange(100, 1101, 1)*units.hPa):
            bottom = top + 50*units.hPa
            selectedData = profileData.where((profileData.LEVEL <= bottom) & (profileData.LEVEL > top), drop=True)
            if len(selectedData.LEVEL.data) <= 1:
                continue
            RHinLayer = mpcalc.mean_pressure_weighted(selectedData.LEVEL, selectedData.RH, height=selectedData.AGL, bottom=selectedData.LEVEL[0], depth=selectedData.LEVEL[0] - selectedData.LEVEL[-1])[0].magnitude
            if RHinLayer >= .8:
                isSaturatedAnywhere = True
                break
    if isSaturatedAnywhere:
        if np.isnan(midLevelThickness) or np.isnan(lowLevelThickness):
            if profileData.TEMP[0] >= 5*units.degC:
                return "sfcRAIN"
            else:
                return "UNKNOWNUNKNOWN"
        elif snowPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SNOW"
        elif unknownPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "UNKNOWN"
        elif snowRainPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SNOW+RAIN"
        elif wintryMixPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "WINTRY MIX"
        elif rainPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "RAIN"
        elif snowSleetPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SNOW+SLEET"
        elif sleetFZRAPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SLEET+FREEZING RAIN"
        elif FZRASleetPoly.contains_points([midLevelThickness, lowLevelThickness]):
            return "FREEZING RAIN+SLEET"
        elif FRZARainPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "FREEZING RAIN"
    else:
        return "NONE"

def plotPsblHazType(profileData, ax, precipType):
    if precipType == "NONE":
        suffix = "(No saturated layers)"
        precipcolor = "peru"
    elif precipType == "SNOW":
        suffix = "(See partial thickness)"
        precipcolor = "cornflowerblue"
    elif precipType == "UNKNOWN":
        suffix = "(See partial thickness)"
        precipcolor = "purple"
    elif precipType == "UNKNOWNUNKNOWN":
        precipType = "UNKNOWN"
        suffix = "(Partial thickness\nunavailable)"
        precipcolor = "black"
    elif precipType == "SNOW+RAIN":
        suffix = "(See partial thickness)"
        precipcolor = "mediumturquoise"
    elif precipType == "WINTRY MIX":
        suffix = "(See partial thickness)"
        precipcolor = "mediumslateblue"
    elif precipType == "RAIN":
        suffix = "(See partial thickness)"
        precipcolor = "limegreen"
    elif precipType == "sfcRAIN":
        precipType = "RAIN"
        suffix = "(Based on\n sfc temp)"
        precipcolor = "limegreen"
    elif precipType == "SNOW+SLEET":
        suffix = "(See partial thickness)"
        precipcolor = "slategray"
    elif "FREEZING RAIN" in precipType:
        suffix = "(See partial thickness)"
        precipcolor = "darkorange"
    else:
        suffix = "(See partial thickness)"
        precipcolor = "violet"
    ax.text(0.5, 0.05, "Precip Type:\n"+precipType+"\n"+suffix, color=precipcolor, ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=10)
    shearms = mpcalc.wind_speed(*profileData.sfc_to_six_shear).to(units.meter/units.second).magnitude
    if shearms > 27:
        shearms = 27
    elif shearms < 7:
        shearms = 7.
    mostUnstableMixingRatio = mpcalc.mixing_ratio_from_relative_humidity(profileData.mu_initPressure, profileData.mu_initTemp, mpcalc.relative_humidity_from_dewpoint(profileData.mu_initTemp, profileData.mu_initDewpoint)).to(units.gram/units.kilogram).magnitude
    if mostUnstableMixingRatio > 13.6:
        mostUnstableMixingRatio = 13.6
    elif mostUnstableMixingRatio < 11:
        mostUnstableMixingRatio = 11
    tempAt500 = profileData.interp(LEVEL_unitless=500).TEMP.data * profileData.TEMP.data.units
    if tempAt500 > -5.5 * units.degC:
        tempAt500 = -5.5 * units.degC
    ship = -1 * profileData.muCAPE * mostUnstableMixingRatio * profileData.five_to_seven_LR * tempAt500 * shearms / 42000000
    if profileData.muCAPE < 1300 * units.joule/units.kilogram:
        ship = ship*(profileData.muCAPE.to(units.joule/units.kilogram).magnitude/1300.)
    if profileData.five_to_seven_LR < 5.8 * units.delta_degree_Celsius/units.km:
        ship = ship*(profileData.five_to_seven_LR.to(units.delta_degree_Celsius/units.km).magnitude/5.8)
    if profileData.freezing_level_agl < 2400 * units.meter:
        ship = ship * (profileData.freezingLevel/2400.)
    ship = ship.magnitude
    if np.isnan(ship):
        ship = 0
    fourToSix = profileData.where(profileData.AGL <= 6000 * units.meter, drop=True).where(profileData.AGL >= 4000 * units.meter, drop=True)
    if profileData.favored_motion is not None:
        favored = profileData.attrs["bunkers_"+profileData.favored_motion]
    else:
        favored = np.array([np.nan, np.nan]) * units.kt
    if len(fourToSix.LEVEL.data) > 0:
        fourToSixMW = mpcalc.mean_pressure_weighted(fourToSix.LEVEL, fourToSix.u, fourToSix.v, height=fourToSix.AGL, bottom=fourToSix.LEVEL[0], depth=(fourToSix.LEVEL[0] - fourToSix.LEVEL[-1]))
        fourToSixSRWu, fourToSixSRWv = (fourToSixMW[0]-favored[0]), (fourToSixMW[1]-favored[1])
        fourToSixSRW = mpcalc.wind_speed(fourToSixSRWu, fourToSixSRWv)
    else:
        fourToSixMW = np.array([np.nan, np.nan]) * units.kt
        fourToSixSRWu = np.nan * units.kt
        fourToSixSRWv = np.nan * units.kt
        fourToSixSRW = np.nan * units.kt
    bulkShear8 = mpcalc.wind_speed(*profileData.sfc_to_eight_shear)
    if profileData.effective_stp >= 3 and profileData.fixed_stp >= 3 and profileData.favored1kmSRH >= 200 * units.meter**2 / units.second**2 and profileData.attrs[profileData.favored_motion+"_SRH"] >= 200 * units.meter**2 / units.second**2 and fourToSixSRW >= 15*units.kt and bulkShear8 > 45 * units.kt and profileData.sbLCL_agl < 1000 * units.meter and profileData.mlLCL_agl < 1200 * units.meter and profileData.sfc_to_one_LR >= 5 *units.delta_degree_Celsius/units.km and profileData.mlCINH > -50*units.joule/units.kilogram and profileData.sbCAPE.to(units.joule/units.kilogram).magnitude == profileData.muCAPE.to(units.joule/units.kilogram).magnitude:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nPDS TOR", color="magenta", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if profileData.effective_stp >= 3 or profileData.fixed_stp >= 4:
        if profileData.mlCINH >= -125 * units.joule/units.kilogram:
            if profileData.sbCAPE.to(units.joule/units.kilogram).magnitude == profileData.muCAPE.to(units.joule/units.kilogram).magnitude:
                ax.text(0.5, 0.9, "Possible\nHazard Type:\nTOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
                return
    if profileData.effective_stp >= 1 or profileData.fixed_stp >= 1:
        if fourToSixSRW >= 15 * units.kt or bulkShear8 > 40 * units.kt:
            if profileData.mlCINH >= -50 * units.joule/units.kilogram and profileData.sbCAPE.to(units.joule/units.kilogram).magnitude == profileData.muCAPE.to(units.joule/units.kilogram).magnitude:
                ax.text(0.5, 0.9, "Possible\nHazard Type:\nTOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
                return
    if profileData.effective_stp >= 1 or profileData.fixed_stp >= 1:
        if np.mean([profileData.LL_RH.magnitude, profileData.ML_RH.magnitude]) >= 0.6 and profileData.sfc_to_one_LR >= 5*units.delta_degree_Celsius/units.km and profileData.mlCINH > -50*units.joule/units.kilogram and profileData.sbCAPE.to(units.joule/units.kilogram).magnitude == profileData.muCAPE.to(units.joule/units.kilogram).magnitude:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nTOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if profileData.effective_stp >= 1 or profileData.fixed_stp >= 1:
        if profileData.mlCINH > -150 * units.joule/units.kilogram and profileData.sbCAPE.to(units.joule/units.kilogram).magnitude == profileData.muCAPE.to(units.joule/units.kilogram).magnitude:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL TOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if profileData.mlCINH > -50 *units.joule/units.kilogram and profileData.muCAPE == profileData.sbCAPE:
        if profileData.effective_stp >= 1 and profileData.attrs[profileData.favored_motion+"_SRH"] >= 150 * units.meter**2 / units.second**2:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL TOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
        if profileData.fixed_stp >= 0.5 and profileData.favored1kmSRH >= 150 * units.meter**2 / units.second**2:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL TOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if profileData.fixed_stp >= 1 or profileData.scp >= 4 or profileData.effective_stp >= 1:
        if profileData.muCINH > -50 * units.joule/units.kilogram:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nSVR", color="y", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if profileData.muCINH > -50 *units.joule/units.kilogram:
        if ship >=1 and profileData.scp >= 2:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nSVR", color="y", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    
    if (profileData.scp >= 2 and ship >= 1) or profileData.dcape >= 750 * units.joule/units.kilogram and profileData.muCINH >= 50 * units.joule/units.kilogram:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nSVR", color="y", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if profileData.muCINH > -75 * units.joule/units.kilogram:
        if ship >= 0.5 or profileData.scp >= 0.5:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL SVR", color="dodgerblue", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if profileData.pwat >= 2 * units.inch and mpcalc.wind_speed(*profileData.corfidi_up) <= 25 * units.kt:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nFLASH FLOOD", color="forestgreen", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if mpcalc.wind_speed(profileData.u.data[0], profileData.v.data[0]) > 35 * units.mile/units.hour and "SNOW" in precipType:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nBLIZZARD", color="darkblue", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if mpcalc.apparent_temperature(profileData.TEMP.data[0], mpcalc.relative_humidity_from_dewpoint(profileData.TEMP.data[0], profileData.DWPT.data[0]), mpcalc.wind_speed(profileData.u.data[0], profileData.v.data[0]), mask_undefined=False) > 105 * units.degF:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nEXCESSIVE HEAT", color="orangered", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if mpcalc.apparent_temperature(profileData.TEMP.data[0], mpcalc.relative_humidity_from_dewpoint(profileData.TEMP.data[0], profileData.DWPT.data[0]), mpcalc.wind_speed(profileData.u.data[0], profileData.v.data[0]), mask_undefined=False) < -20 * units.degF:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nWIND CHILL", color="blue", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    else:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nNONE", color="black", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return

def plotHodograph(profileData, ax):
    hodoPlot = plots.Hodograph(ax, component_range=200)
    hodoPlot.add_grid(increment=10)
    [ax.text(0, i, f"{i} kt", color="gray", clip_on=True, zorder=6) for i in np.arange(-200, 201, 10).astype(int)]
    [ax.text(i, 0, f"{i} kt", color="gray", clip_on=True, zorder=6) for i in np.arange(-200, 201, 10).astype(int)]
    profileData.u.data = profileData.u.data.to(units.kt)
    profileData.v.data = profileData.v.data.to(units.kt)
    intervals = [0]
    colors = ["fuchsia"]
    if np.nanmax(profileData.AGL.data) < 1 * units.km:
        intervals.append(np.nanmax(profileData.AGL.data.to(units.km).magnitude)-.001)
        intervals.append(np.nanmax(profileData.AGL.data.to(units.km).magnitude))
        colors.append(["white"])
    elif np.nanmax(profileData.AGL.data) >= 1 * units.km:
        intervals.append(1)
        intervals.append(3)
        colors.append("firebrick")
        if np.nanmax(profileData.AGL.data) >= 3 * units.km:
            intervals.append(6)
            colors.append("limegreen")
            if np.nanmax(profileData.AGL.data) >= 6 * units.km:
                intervals.append(9)
                colors.append("goldenrod")
                if np.nanmax(profileData.AGL.data) >= 9 * units.km:
                    intervals.append(12)
                    colors.append("darkturquoise")
                    if np.nanmax(profileData.AGL.data) >= 12 * units.km:
                        intervals.append(15)
                        colors.append("darkturquoise")
    intervals = np.array(intervals)*units.km
    hodoPlot.plot_colormapped(profileData.u.data.magnitude, profileData.v.data.magnitude, profileData.AGL.data, colors=colors, intervals=intervals)
    maxKm = np.nanmax(profileData.AGL.data.magnitude) // 1000
    kmToLabel = list(np.arange(1, np.min([maxKm+1, 7])))
    ax.scatter(profileData.u.data[0], profileData.v.data[0], color="black", marker="o", s=125, zorder=4)
    ax.scatter(profileData.u.data[0], profileData.v.data[0], color="white", marker="o", s=100, zorder=4)
    ax.annotate("0", [profileData.u.data[0], profileData.v.data[0]], color="black", clip_on=True, zorder=5, ha="center", va="center")
    if maxKm >= 1:
        kmToLabel.extend([0.5])
        if maxKm >=9:
            kmToLabel.extend([9])
            if maxKm >= 12:
                kmToLabel.extend([12])
                if maxKm >= 15:
                    kmToLabel.extend([15])
    for i in kmToLabel:
        selectedIdx = np.argmin(np.abs(profileData.AGL.data.to(units.meter).magnitude - i*1000))
        closestValueToTarget = profileData.isel(LEVEL_unitless=selectedIdx)
        ax.scatter(closestValueToTarget.u.data.magnitude, closestValueToTarget.v.data.magnitude, color="black", marker="o", s=125, zorder=4)
        ax.scatter(closestValueToTarget.u.data.magnitude, closestValueToTarget.v.data.magnitude, color="white", marker="o", s=100, zorder=4)
        textToPlot = str(int(i)) if i != 0.5 else ".5"
        ax.annotate(textToPlot, [closestValueToTarget.u.data.magnitude, closestValueToTarget.v.data.magnitude], color="black", clip_on=True, zorder=5, ha="center", va="center")
    lowest500m = profileData.where(profileData.AGL <= 500 * units.m).mean()
    rm = profileData.attrs["bunkers_RM"]
    rmMag, rmDir = mpcalc.wind_speed(*rm), mpcalc.wind_direction(*rm)
    lm = profileData.attrs["bunkers_LM"]
    lmMag, lmDir = mpcalc.wind_speed(*lm), mpcalc.wind_direction(*lm)
    if profileData.favored_motion == "RM":
        ax.scatter(rm[0], rm[1], color="white", edgecolor="red", marker="o", s=50, zorder=4, label=f"Bunkers Right [{rmMag.magnitude:.1f} kt, {rmDir.magnitude:.1f}°]")
        ax.scatter(lm[0], lm[1], color="white", edgecolor="blue", marker="o", s=50, zorder=4, alpha=0.5, label=f"Bunkers Left [{lmMag.magnitude:.1f} kt, {lmDir.magnitude:.1f}°]")
        favoredU, favoredV = rm[0].to(units.kt), rm[1].to(units.kt)
    else:
        ax.scatter(rm[0], rm[1], color="white", edgecolor="red", marker="o", s=50, zorder=4, alpha=0.5, label=f"Bunkers Right [{rmMag.magnitude:.1f} kt, {rmDir.magnitude:.1f}°]")
        ax.scatter(lm[0], lm[1], color="white", edgecolor="blue", marker="o", s=50, zorder=4, label=f"Bunkers Left [{lmMag.magnitude:.1f} kt, {lmDir.magnitude:.1f}°]")
        favoredU, favoredV = lm[0].to(units.kt), lm[1].to(units.kt)
    dtmU = (favoredU + lowest500m.u.data)/2
    dtmV = (favoredV + lowest500m.v.data)/2
    dtmMag, dtmDir = mpcalc.wind_speed(dtmU, dtmV), mpcalc.wind_direction(dtmU, dtmV)
    ax.scatter(dtmU.magnitude, dtmV.magnitude, color="white", edgecolor="darkviolet", marker="v", s=50, zorder=4, label=f"Deviant Tornado Motion [{dtmMag.magnitude:.1f} kt, {dtmDir.magnitude:.1f}°]")
    mwMag, mwDir = mpcalc.wind_speed(profileData.zeroToSixMean[0], profileData.zeroToSixMean[1]), mpcalc.wind_direction(profileData.zeroToSixMean[0], profileData.zeroToSixMean[1])
    ax.scatter(profileData.zeroToSixMean[0], profileData.zeroToSixMean[1], color="white", edgecolor="sienna", marker="s", s=50, zorder=4, label=f"0->6km Mean Wind [{mwMag.magnitude:.1f} kt, {mwDir.magnitude:.1f}°]")
    
    corfidiUpshearMag, corfidiUpshearDir = mpcalc.wind_speed(*profileData.corfidi_up).to(units.kt), mpcalc.wind_direction(*profileData.corfidi_up)
    ax.scatter(profileData.corfidi_up[0].to(units.knot).magnitude, profileData.corfidi_up[1].to(units.knot).magnitude, color="limegreen", marker="2", s=50, zorder=4, label=f"Corfidi Upshear [{corfidiUpshearMag.magnitude:.1f} kt, {corfidiUpshearDir.magnitude:.1f}°]")
    corfidiDownshearMag, corfidiDownshearDir = mpcalc.wind_speed(*profileData.corfidi_down).to(units.kt), mpcalc.wind_direction(*profileData.corfidi_down)
    ax.scatter(profileData.corfidi_down[0].to(units.knot).magnitude, profileData.corfidi_down[1].to(units.knot).magnitude, color="darkgreen", marker="1", s=50, zorder=4, label=f"Corfidi Downshear [{corfidiDownshearMag.magnitude:.1f} kt, {corfidiDownshearDir.magnitude:.1f}°]")
    if not np.isnan(profileData.inflowBottom):
        eil = profileData.where(profileData.LEVEL <= profileData.inflowBottom, drop=True).where(profileData.LEVEL >= profileData.inflowTop, drop=True)
        critAngle = mpcalc.critical_angle(profileData.LEVEL, profileData.u, profileData.v, profileData.HGHT, favoredU, favoredV)
        ax.plot([favoredU.magnitude, eil.u.data.to(units.knot)[0].magnitude], [favoredV.magnitude, eil.v.data.to(units.knot)[0].magnitude], color="teal", linestyle="--", linewidth=0.5, zorder=1, label="Eff. Inflow [Crit. Angle: {:.1f}°]".format(critAngle.magnitude))
        ax.plot([favoredU.magnitude, eil.u.data.to(units.knot)[-1].magnitude], [favoredV.magnitude, eil.v.data.to(units.knot)[-1].magnitude], color="teal", linestyle="--", linewidth=0.5, zorder=1)
        closestValueToTarget = profileData.isel(LEVEL_unitless=np.argmin(np.abs(profileData.AGL.data.to(units.meter).magnitude - 500)))
        ax.plot([closestValueToTarget.u.data.magnitude, eil.u.data.to(units.knot)[0].magnitude], [closestValueToTarget.v.data.magnitude, eil.v.data.to(units.knot)[0].magnitude], color="mediumvioletred", linestyle="--", linewidth=0.5, zorder=1)
    

    profileUnder12km = profileData.where(profileData.AGL <= 12000 * units.meter, drop=True)


    ax.set_xlim(np.nanmin([np.nanmin(profileUnder12km.u.data.to(units.kt).magnitude), profileData.corfidi_up[0].magnitude, profileData.corfidi_down[0].magnitude, dtmU.magnitude, profileData.zeroToSixMean[0].magnitude, rm[0].magnitude, lm[0].magnitude])-2.5, np.nanmax([np.nanmax(profileUnder12km.u.data.to(units.kt).magnitude), profileData.corfidi_up[0].magnitude, profileData.corfidi_down[0].magnitude, dtmU.magnitude, profileData.zeroToSixMean[0].magnitude, rm[0].magnitude, lm[0].magnitude])+2.5)
    ax.set_ylim(np.nanmin([np.nanmin(profileUnder12km.v.data.to(units.kt).magnitude), profileData.corfidi_up[1].magnitude, profileData.corfidi_down[1].magnitude, dtmV.magnitude, profileData.zeroToSixMean[1].magnitude, rm[1].magnitude, lm[1].magnitude])-2.5, np.nanmax([np.nanmax(profileUnder12km.v.data.to(units.kt).magnitude), profileData.corfidi_up[1].magnitude, profileData.corfidi_down[1].magnitude, dtmV.magnitude, profileData.zeroToSixMean[1].magnitude, rm[1].magnitude, lm[1].magnitude])+2.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.legend()

def plotThermalWind(profileData, ax, latitude):
    if latitude is not None:
        uflip = np.flip(profileData.u.data)
        vflip = np.flip(profileData.v.data)
        pflip = np.flip(profileData.LEVEL.data)
        dvdP = np.flip(mpcalc.first_derivative(vflip, x=pflip))
        dudP = np.flip(mpcalc.first_derivative(uflip, x=pflip))

        f = mpcalc.coriolis_parameter(latitude)
        R = constants.dry_air_gas_constant.to_base_units()
        
        dTdx = - (f/R) * (profileData.LEVEL.data)* dvdP
        dTdy = (f/R) * (profileData.LEVEL.data) * dudP
        dTdt = (- profileData.u.data * dTdx - profileData.v.data * dTdy).to(units.kelvin / units.hour)
        ax.fill_betweenx(profileData.LEVEL.data.to(units.hPa), dTdt, 0, where=dTdt > 0, color="red", alpha=0.5, zorder=2)
        ax.fill_betweenx(profileData.LEVEL.data.to(units.hPa), dTdt, 0, where=dTdt < 0, color="blue", alpha=0.5, zorder=2)

        minimas = (np.diff(np.sign(np.diff(dTdt))) > 0).nonzero()[0] + 1 
        maximas = (np.diff(np.sign(np.diff(dTdt))) < 0).nonzero()[0] + 1
        minmaxes = np.sort(np.append(minimas, maximas))
        lastMaxText = None
        lastMaxLvl = (np.nanmax(profileData.LEVEL.data.to(units.hPa).magnitude)+100) * units.hPa
        lastMaxValue = 0
        lastMinText = None
        lastMinLvl = (np.nanmax(profileData.LEVEL.data.to(units.hPa).magnitude)+100) * units.hPa
        lastMinValue = 0
        for minOrMax in minmaxes:
            value = dTdt[minOrMax] * (units.degK / units.hour)
            lvl = profileData.LEVEL.data[minOrMax]
            if lvl < 100 * units.hPa:
                break
            if value < 0:
                if lastMinText is not None:
                    if np.abs(lvl - lastMinLvl) < 20  * units.hPa:
                        if np.abs(value) > np.abs(lastMinValue):
                            lastMinText.remove()
                        else:
                            continue
                lastMinText = ax.text(np.nanmin(dTdt), lvl.to(units.hPa).magnitude, f"{value.magnitude:.1f}", color="blue",  ha="left", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], clip_on=False, alpha=0.5, zorder=3)
                lastMinLvl = lvl
                lastMinValue = value
            else:
                if lastMaxText is not None:
                    if np.abs(lvl - lastMaxLvl) < 20 * units.hPa:
                        if np.abs(value) > np.abs(lastMaxValue):
                            lastMaxText.remove()
                        else:
                            continue
                lastMaxText = ax.text(np.nanmax(dTdt), lvl.to(units.hPa).magnitude, f"{value.magnitude:.1f}", color="red",  ha="right", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], clip_on=False, alpha=0.5, zorder=3)
                lastMaxLvl = lvl
                lastMaxValue = value
        ax.set_xlabel("Temp. Adv.\n(K/hr)")
        ax.set_xlim(np.nanmin(dTdt), np.nanmax(dTdt))
    else:
        ax.set_xlabel("RH %")
        ax.set_xlim(0, 100)
    for top in np.arange(100, 1101, 100) * units.hPa:
        bottom = top + 100 * units.hPa
        selectedData = profileData.where((profileData.LEVEL <= bottom) & (profileData.LEVEL > top), drop=True)
        if len(selectedData.LEVEL.data) <= 0:
            continue
        RHinLayer = mpcalc.mean_pressure_weighted(selectedData.LEVEL, selectedData.RH, height=selectedData.AGL, bottom=selectedData.LEVEL[0], depth=(selectedData.LEVEL[0] - selectedData.LEVEL[-1]))[0].magnitude
        ax.plot([RHinLayer, RHinLayer], [top.magnitude, bottom.magnitude], color="forestgreen", linewidth=1, zorder=1, transform=ax.get_yaxis_transform())
        ax.fill_betweenx([bottom.magnitude, top.magnitude], RHinLayer, 0, color="limegreen", alpha=0.5, zorder=1, transform=ax.get_yaxis_transform())
        ax.text(RHinLayer, (top+bottom).magnitude/2, f"{int(RHinLayer*100)}%", color="forestgreen",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], clip_on=False, alpha=0.5, zorder=3, transform=ax.get_yaxis_transform())
    ax.hlines(range(100, 1001, 100), xmin=-100, xmax=100, color="gray", linewidth=0.5, zorder=1)
    ax.tick_params(which="both", labelleft=False, direction="in", pad=-9)
    ax.set_ylabel("")
    return profileData

def plotSkewT(profileData, skew, parcelType="sb"):
    # Plot data
    skew.plot(profileData.LEVEL, profileData.TEMP.data.to(units.degC), "red", zorder=5)
    skew.plot(profileData.LEVEL, profileData.DWPT.data.to(units.degC), "lime", zorder=4)
    skew.plot(profileData.LEVEL, profileData.virtT.data.to(units.degC), "red", linestyle=":", zorder=4)
    skew.plot(profileData.LEVEL, profileData.wetbulb.data.to(units.degC), "cyan", linewidth=0.5, zorder=3)
    mask = mpcalc.resample_nn_1d(profileData.LEVEL.data.to(units.hPa).magnitude,  np.logspace(4, 2))
    skew.plot_barbs(profileData.LEVEL.data[mask], profileData.u.data.to(units.kt)[mask], profileData.v.data.to(units.kt)[mask])
    xlimmin = 10*(np.nanmin(profileData.TEMP.data.to(units.degC).magnitude) // 10)+10
    xlimmax = (10*np.nanmax(profileData.TEMP.data.to(units.degC).magnitude) // 10)+10
    if xlimmax - xlimmin < 110:
        xlimmin = xlimmax - 110
    skew.ax.set_xlim(xlimmin, xlimmax)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.set_ylim(np.nanmax(profileData.LEVEL.data.magnitude)+5, 100)
    skew.ax.tick_params(which="both", direction="in")
    skew.ax.tick_params(axis="x", direction="in", pad=-9)
    skew.plot_dry_adiabats(color="gray", linewidths=0.2, zorder=1)
    skew.plot_moist_adiabats(color="gray", linewidths=0.2, zorder=1)
    skew.ax.vlines([-12, -17], 100, np.nanmax(profileData.LEVEL), color="blue", linestyle="--", linewidth=0.5, zorder=1)

    skew.ax.text(profileData.wetbulb.data.to(units.degC).magnitude[0], -0.012, f"{int((profileData.wetbulb.data[0]).to(units.degF).magnitude)} °F", color="cyan",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_xaxis_transform())
    skew.ax.text(profileData.DWPT.data.to(units.degC).magnitude[0]-5, -0.012, f"{str(int((profileData.DWPT.data[0]).to(units.degF).magnitude))} °F", color="lime",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_xaxis_transform(), alpha=0.5)
    skew.ax.text(profileData.TEMP.data.to(units.degC).magnitude[0]+5, -0.012, f"{int((profileData.TEMP.data[0]).to(units.degF).magnitude)} °F", color="red",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_xaxis_transform(), alpha=0.5)

    if not np.isnan(profileData.dcape_levels.data[0]):
        skew.plot(profileData.dcape_levels, profileData.dcape_profile, "rebeccapurple", linestyle="--", markersize=3, zorder=6)


    kmToInterp = np.arange(0, 15001, 1000)
    interpPressures = np.interp(kmToInterp, profileData.AGL, profileData.LEVEL)
    colorsAndKilometers = {
            (0, 1): "fuchsia",
            (1, 3): "firebrick",
            (3, 6): "limegreen",
            (6, 9): "goldenrod",
            (9, 12): "darkturquoise",
            (12, 15): "darkturquoise"
    }
    for indices, color in reversed(colorsAndKilometers.items()):
        bot, top = indices
        skew.ax.add_patch(Polygon([(0, interpPressures[bot]), (0.01, interpPressures[bot]), (0.01, interpPressures[top]), (0, interpPressures[top])], closed=True, color=color, alpha=0.5, zorder=1, transform=skew.ax.get_yaxis_transform()))
    for i in [0, 1, 2, 3, 4, 5, 6, 9, 12, 15]:
        if np.nanmax(profileData.AGL.data) < i * units.km:
            skew.ax.text(0.01, np.nanmin(interpPressures), f"{np.nanmax(profileData.AGL.data).to(units.km).magnitude:.3f} km: {np.nanmin(profileData.LEVEL.data).to(units.hPa).magnitude:.1f} hPa", color="black", transform=skew.ax.get_yaxis_transform())
            break
        if i == 0:
            skew.ax.text(0.01, interpPressures[i]-5, f"SFC: {interpPressures[i]:.1f} hPa", color="black", transform=skew.ax.get_yaxis_transform())
        else:
            skew.ax.text(0.01, interpPressures[i], f"{str(int(i))} km: {interpPressures[i]:.1f} hPa", color="black", transform=skew.ax.get_yaxis_transform())
            skew.ax.plot([0, 0.05], [interpPressures[i], interpPressures[i]], color="gray", linewidth=.75, transform=skew.ax.get_yaxis_transform())
    
    if not np.isnan(profileData.inflowBottom):
        eilData = profileData.where((profileData.LEVEL <= profileData.inflowBottom) & (profileData.LEVEL >= profileData.inflowTop), drop=True)
        eilRH = int(eilData.RH.mean() * 100)

        skew.ax.text(0.16, profileData.inflowTop.to(units.hPa).magnitude, f"Effective Inflow Layer\n{profileData.inflowBottom.to(units.hPa).magnitude:.1f} - {profileData.inflowTop.to(units.hPa).magnitude:.1f} hPa\nAGL: {int(profileData.inflowBottom_agl.to(units.meter).magnitude)} - {int(profileData.inflowTop_agl.to(units.meter).magnitude)} m\nRH: {eilRH}%", color="teal",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)
        skew.ax.plot([0.15, 0.15], [profileData.inflowBottom.to(units.hPa).magnitude, profileData.inflowTop.to(units.hPa).magnitude], color="teal", linewidth=1.25, transform=skew.ax.get_yaxis_transform())
        skew.ax.plot([0, 0.25], [profileData.inflowBottom.to(units.hPa).magnitude, profileData.inflowBottom.to(units.hPa).magnitude], color="teal", linewidth=1.25, transform=skew.ax.get_yaxis_transform())
        skew.ax.plot([0, 0.25], [profileData.inflowTop.to(units.hPa).magnitude, profileData.inflowTop.to(units.hPa).magnitude], color="teal", linewidth=1.25, transform=skew.ax.get_yaxis_transform())

    skew.plot(profileData.LEVEL, profileData[parcelType+"ParcelPath"], "black", linewidth=1, zorder=6, linestyle="dashdot")
    skew.ax.plot([0, .95], [profileData.attrs[parcelType+"LCL"].magnitude, profileData.attrs[parcelType+"LCL"].magnitude], color="mediumseagreen", linewidth=1, transform=skew.ax.get_yaxis_transform())
    skew.ax.text(0.875, profileData.attrs[parcelType+"LCL"].magnitude, f"LCL: {profileData.attrs[parcelType+'LCL'].magnitude:.1f} hPa", color="mediumseagreen",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
    skew.ax.plot([0, .95], [profileData.attrs[parcelType+"LFC"].magnitude, profileData.attrs[parcelType+"LFC"].magnitude], color="darkgoldenrod", linewidth=1, transform=skew.ax.get_yaxis_transform())
    skew.ax.text(0.875, profileData.attrs[parcelType+"LFC"].magnitude, f"LFC: {profileData.attrs[parcelType+'LFC'].magnitude:.1f} hPa", color="darkgoldenrod",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
    skew.ax.plot([0, .95], [profileData.attrs[parcelType+"EL"].magnitude, profileData.attrs[parcelType+"EL"].magnitude], color="mediumpurple", linewidth=1, transform=skew.ax.get_yaxis_transform())
    skew.ax.text(0.875, profileData.attrs[parcelType+"EL"].magnitude, f"EL: {profileData.attrs[parcelType+'EL'].magnitude:.1f} hPa", color="mediumpurple",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
   

    dgzsData = profileData.where(profileData.TEMP <= -12*units.degC, drop=True).where(profileData.TEMP >= -17*units.degC, drop=True)
    
    if len(dgzsData.LEVEL.data) > 0:
        listOfDGZs = []
        dgzBottom = dgzsData.LEVEL.data[0]
        for i in range(1, len(dgzsData.LEVEL.data)):
            if dgzsData.LEVEL[i-1] - dgzsData.LEVEL[i] >= 2 * units.hPa:
                dgzTop = dgzsData.LEVEL.data[i-1]
                listOfDGZs.append((dgzBottom, dgzTop))
                dgzBottom = dgzsData.LEVEL.data[i]
        dgzTop = dgzsData.LEVEL.data[-1]
        listOfDGZs.append((dgzBottom, dgzTop))
        for dgz in listOfDGZs:
            dgzData = profileData.where((profileData.LEVEL <= dgz[0]), drop=True).where((profileData.LEVEL >= dgz[1]), drop=True)
            skew.shade_area(dgzData.LEVEL.data.to(units.hPa).magnitude, dgzData.TEMP.data.to(units.degC).magnitude, dgzData.DWPT.data.to(units.degC).magnitude, color="blue", alpha=0.2, zorder=6)
            skew.ax.plot([0, 1], [dgzData.LEVEL[-1].data.to(units.hPa).magnitude, dgzData.LEVEL[-1].data.to(units.hPa).magnitude], color="blue", linewidth=.75, transform=skew.ax.get_yaxis_transform())
            skew.ax.plot([0, 1], [dgzData.LEVEL[0].data.to(units.hPa).magnitude, dgzData.LEVEL[0].data.to(units.hPa).magnitude], color="blue", linewidth=.75, transform=skew.ax.get_yaxis_transform())
            
            dgzData.LEVEL.data.to(units.hPa).mean().magnitude
            f"Dendritic Growth Zone\n{dgzData.LEVEL[0].data.to(units.hPa):.1f}"
            f"{dgzData.LEVEL[-1].data.to(units.hPa):.1f}hPa"
            f"AGL: {int(dgzData.AGL.data[0].to(units.meter).magnitude)}"
            f"{int(dgzData.AGL.data[-1].to(units.meter).magnitude)}m"
            f"RH: {int(dgzData.RH.data.mean().magnitude*100)}%"
            
            skew.ax.text(0.15, dgzData.LEVEL.data.to(units.hPa).mean().magnitude, f"Dendritic Growth Zone\n{dgzData.LEVEL[0].data.to(units.hPa).magnitude:.1f} - {dgzData.LEVEL[-1].data.to(units.hPa).magnitude:.1f}hPa\nAGL: {int(dgzData.AGL.data[0].to(units.meter).magnitude)} - {int(dgzData.AGL.data[-1].to(units.meter).magnitude)}m\nRH: {int(dgzData.RH.data.mean().magnitude*100)}%", color="blue",  ha="left", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)
    


def plotSounding(profileData, outputPath, icao, time, soundingType="Observed"):
    fig = plt.figure()
    tax = fig.add_axes([1/20, 14/16, 18/20, 1/16])
    if not np.isnan(profileData.LAT) and not np.isnan(profileData.LON):
        try:
            if len(profileData.LAT) == 1:
                groundLat = profileData.LAT
                groundLon = profileData.LON
            else:
                groundLat = profileData.LAT[0]
                groundLon = profileData.LON[0]
        except TypeError:
            groundLat = profileData.LAT
            groundLon = profileData.LON
        tax.text(0.5, 0.5, f"{soundingType} Sounding -- {time.strftime('%H:%M UTC %d %b %Y')} -- {icao} ({groundLat.magnitude:.2f}, {groundLon.magnitude:.2f})", ha="center", va="center", transform=tax.transAxes)
    else:
        groundLat = None
        groundLon = None
        tax.text(0.5, 0.5, f"{soundingType} Sounding -- {time.strftime('%H:%M UTC %d %b %Y')} -- {icao}", ha="center", va="center", transform=tax.transAxes)
    tax.axis("off")
    skew = plots.SkewT(fig, rect=[1/20, 4/16, 10/20, 10/16])
    plotSkewT(profileData, skew)
    skew.ax.patch.set_alpha(0)

    thermalWindAx = fig.add_axes([11/20, 4/16, 1/20, 10/16])
    thermalWindAx.set_yscale("log")
    thermalWindAx.set_ylim(skew.ax.get_ylim())
    if groundLat is not None:
        thermalWindAx.text(0.5, 0.95, "Thermal Wind\nRel. Humidity", ha="center", va="center", fontsize=9, transform=thermalWindAx.transAxes)
    else:
        thermalWindAx.text(0.5, 0.95, "Rel. Humidity", ha="center", va="center", fontsize=9, transform=thermalWindAx.transAxes)
    plotThermalWind(profileData, thermalWindAx, groundLat)
    thermalWindAx.patch.set_alpha(0)
    
    hodoAx = fig.add_axes([12/20, 9/16, 7/20, 5/16])
    plotHodograph(profileData, hodoAx)
    hodoAx.patch.set_alpha(0)
    
    partialThicknessAx = fig.add_axes([14/20, 7/16, 2/20, 2/16])
    precipType = plotPartialThickness(profileData, partialThicknessAx)
    
    psblHazTypeAx = fig.add_axes([12/20, 7/16, 2/20, 2/16])
    psblHazTypeAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotPsblHazType(profileData, psblHazTypeAx, precipType)



    mapAx = fig.add_axes([16/20, 7/16, 3/20, 2/16], projection=ccrs.PlateCarree())
    if groundLat is not None and groundLon is not None:
        mapAx.scatter(groundLon, groundLat, transform=ccrs.PlateCarree(), color="black", marker="*")
        mapAx.add_feature(cfeat.STATES.with_scale("50m"))
        mapAx.add_feature(plots.USCOUNTIES.with_scale("5m"), edgecolor="gray", linewidth=0.25)
    else:
        mapAx.text(0.5, 0.5, "Location not available", ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=mapAx.transAxes)
    mapAx.add_feature(cfeat.COASTLINE.with_scale("50m"))
    
    thermodynamicsAx = fig.add_axes([1/20, 1/16, 9/20, 3/16])
    thermodynamicsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    thermodynamicsAx.spines[['right']].set_visible(False)
    plotThermoynamics(profileData, thermodynamicsAx)
    thermodynamicsAx.patch.set_alpha(0)

    paramsAx = fig.add_axes([10/20, 1/16, 2/20, 3/16])
    paramsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotParams(profileData, paramsAx)
    paramsAx.patch.set_alpha(0)

    dynamicsAx = fig.add_axes([12/20, 1/16, 7/20, 6/16])
    dynamicsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotDynamics(profileData, dynamicsAx)

    px = 1/plt.rcParams["figure.dpi"]
    fig.set_size_inches(1920*px, 1080*px)
    width_unit = skew.ax.get_position().width / 10
    height_unit = skew.ax.get_position().height / 10
    tax.set_position([1*width_unit, 14*height_unit, 18*width_unit, 1*height_unit])
    skew.ax.set_position([1*width_unit, 4*height_unit, 10*width_unit, 10*height_unit])
    thermalWindAx.set_position([11*width_unit, 4*height_unit, width_unit, 10*height_unit])
    oldHodoLimits = hodoAx.get_xlim(), hodoAx.get_ylim()
    hodoAx.set_adjustable("datalim")
    hodoAx.set_position([12*width_unit, 9*height_unit, 7*width_unit, 5*height_unit])
    hodoaspect = (hodoAx.get_position().height*1080)/(hodoAx.get_position().width*1920)
    altxmax = ((oldHodoLimits[1][1] - oldHodoLimits[1][0])/hodoaspect)+oldHodoLimits[0][0]
    altymax = ((oldHodoLimits[0][1] - oldHodoLimits[0][0])*hodoaspect)+oldHodoLimits[1][0]
    hodoAx.set_xlim(oldHodoLimits[0][0], np.nanmax([altxmax, oldHodoLimits[0][1]]))
    hodoAx.set_ylim(oldHodoLimits[1][0], np.nanmax([altymax, oldHodoLimits[1][1]]))

    psblHazTypeAx.set_position([12*width_unit, 7*height_unit, 2*width_unit, 2*height_unit])
    partialThicknessAx.set_position([14*width_unit, 7*height_unit, 2*width_unit, 2*height_unit])

    thermodynamicsAx.set_position([1*width_unit, 1*height_unit, 9*width_unit, 3*height_unit])
    paramsAx.set_position([10*width_unit, 1*height_unit, 2*width_unit, 3*height_unit])
    dynamicsAx.set_position([12*width_unit, 1*height_unit, 7*width_unit, 6*height_unit])

    mapAx.set_adjustable("datalim")
    mapAx.set_position([16*width_unit, 7*height_unit, 3*width_unit, 2*height_unit])
    Path(path.dirname(outputPath)).mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath)
    plt.close(fig)


def readACARS(acarsDatasetPath):
    acarsDataset = xr.open_dataset(acarsDatasetPath)
    air = airportsdata.load("IATA")
    for i in range(len(acarsDataset.recNum)):
        thisSounding = acarsDataset.isel(recNum=i)
        thisSoundingICAO = air[bytes(thisSounding.profileAirport.data).decode("utf-8")[:3]]["icao"]
        thisSoundingTime = pd.to_datetime(pd.Timestamp(thisSounding.profileTime.data.item()))
        saveFilePath = path.join(sys.argv[2], thisSoundingICAO, thisSoundingTime.strftime("%Y"), thisSoundingTime.strftime("%m"), thisSoundingTime.strftime("%d"), thisSoundingTime.strftime("%H%M.png"))
        if path.exists(saveFilePath):
            continue
        metarsRaw = requests.get(f"https://www.aviationweather.gov/metar/data?ids={thisSoundingICAO}&format=raw&hours=12&taf=off&layout=on").content.decode("utf-8").split(">\n<code>")
        if "No METAR found for" in metarsRaw:
            continue
        allMetars = None
        for metar in metarsRaw:
            metarStr = metar.replace("</code><br/", "").split(">\n<")[0]
            if metarStr.startswith(thisSoundingICAO):
                metar = parse_metar_to_dataframe(metarStr)
                if allMetars is None:
                    allMetars = metar
                else:
                    allMetars = pd.concat([allMetars, metar])
        if allMetars is None:
            continue
        allMetars["timedelta"] = allMetars["date_time"].apply(lambda x: abs((x - thisSoundingTime).total_seconds()))
        closestMetar = allMetars.sort_values("timedelta").sort_values("timedelta").iloc[0]
        altitudes = [closestMetar["elevation"] * units("m")]
        temperatures = [closestMetar["air_temperature"] * units("degC")]
        dewpoints = [closestMetar["dew_point_temperature"] * units("degC")]
        windDirection = [closestMetar["wind_direction"] * units.deg]
        windSpeed = [closestMetar["wind_speed"] * units("kt")]
        airportBarometer = mpcalc.altimeter_to_station_pressure(closestMetar["altimeter"] * units("inHg"), closestMetar["elevation"] * units("m")).to("hPa")
        pressure = [airportBarometer]
        virtTemp = [mpcalc.virtual_temperature(closestMetar["air_temperature"] * units("degC"), mpcalc.mixing_ratio_from_specific_humidity(mpcalc.specific_humidity_from_dewpoint(airportBarometer, closestMetar["dew_point_temperature"] * units("degC"))))]
        for ii in range(0, len(thisSounding.altitude)):
            if np.isnan(thisSounding.altitude.data[ii]) or np.isnan(thisSounding.temperature.data[ii]):# or np.isnan(thisSounding.dewpoint.data[ii]) or np.isnan(thisSounding.windDir.data[ii]) or np.isnan(thisSounding.windSpeed.data[ii]):
                continue
            if thisSounding.altitude.data[ii] * units.meter in altitudes:
                continue
            thisTemp = thisSounding.temperature.data[ii] * units("K")
            lastTemp = temperatures[-1]
            thisAlt = thisSounding.altitude.data[ii] * units("m")
            lastAlt = altitudes[-1]
            lastPressure = pressure[-1]
            
            thisPressureFactor = (np.abs(lastTemp*lastAlt-thisTemp*thisAlt)/np.abs(lastTemp*(2*lastAlt-thisAlt)-thisTemp*lastAlt))**((9.81*(lastAlt-thisAlt))/(287*(lastTemp-thisTemp))).magnitude
            topPressure = lastPressure*thisPressureFactor**(-1)
            
            if topPressure not in pressure:
                temperatures.append(thisTemp)
                dewpoints.append(thisSounding.dewpoint.data[ii] * units("K"))
                windDirection.append(thisSounding.windDir.data[ii] * units.deg)
                windSpeed.append((thisSounding.windSpeed.data[ii] * units("meter / second")).to("kt"))
                altitudes.append(thisAlt)
                pressure.append(topPressure)
                virtTemp.append(mpcalc.virtual_temperature(thisTemp, mpcalc.mixing_ratio_from_specific_humidity(mpcalc.specific_humidity_from_dewpoint(lastPressure, dewpoints[-1]))))
        
        pressure = np.array([pres.to("hPa").magnitude for pres in pressure])
        temperatures = np.array([temp.to("degC").magnitude for temp in temperatures])
        dewpoints = np.array([dew.to("degC").magnitude for dew in dewpoints])
        windDirection = np.array([windDir.to("deg").magnitude for windDir in windDirection])
        windSpeed = np.array([windSpd.to("kt").magnitude for windSpd in windSpeed])
        altitudes = np.array([alt.to("m").magnitude for alt in altitudes])

        soundingDataFrame = pd.DataFrame({"LEVEL": pressure, "HGHT": altitudes, "TEMP": temperatures, "DWPT": dewpoints, "WDIR": windDirection, "WSPD": windSpeed}).sort_values("LEVEL", ascending=False)
        data = makeSoundingDataset(soundingDataFrame, thisSoundingICAO, thisSoundingTime)
        saveFilePath = path.join(sys.argv[2], thisSoundingICAO, thisSoundingTime.strftime("%Y"), thisSoundingTime.strftime("%m"), thisSoundingTime.strftime("%d"), thisSoundingTime.strftime("%H%M.png"))
        print(thisSoundingICAO, thisSoundingTime)
        plotSounding(data, saveFilePath, thisSoundingICAO, thisSoundingTime, soundingType="ACARS")

def readSharppy(fileName):
    from io import StringIO
    with open(fileName, "r") as f:
        text = f.readlines()
    preamble = text[0]
    whenAndWhere = text[1]
    if whenAndWhere.startswith(" "):
        where = whenAndWhere.split(" ")[1]
    else:
        where = whenAndWhere.split(" ")[0]
    when = dt.strptime(str(dt.utcnow().year)[:2]+whenAndWhere.split(" ")[-1], "%Y%m%d/%H%M\n")
    who = text[2]
    rest = "".join(text[3:])
    data = "".join(rest.split("%RAW%")[1]).split("%END%")[0].replace(" ", "")
    data = pd.read_csv(StringIO(data), sep=",", header=None, names=["LEVEL", "HGHT", "TEMP", "DWPT", "WDIR", "WSPD"])
    data = data.replace(-9999, np.nan)
    for i in range(len(data)):
        if np.isnan(data.iloc[0]["TEMP"]) or np.isnan(data.iloc[0]["DWPT"]) or np.isnan(data.iloc[0]["HGHT"]) or np.isnan(data.iloc[0]["LEVEL"]):
            print(f"Warning: packet {i} is invalid, removing")
            data = data.iloc[1:]
        else:
            break
    data = data.loc[data["LEVEL"] >= 10]
    data = makeSoundingDataset(data, where, when)
    return data, where, when

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: soundingPlot.py <input> <output>")
        exit()
    if not path.exists(sys.argv[1]):
        print("Input file does not exist!")
        exit()
    if sys.argv[1].endswith("acars.nc"):
        readACARS(sys.argv[1])
    else:
        profileData, icao, datetime  = readSharppy(sys.argv[1])
        plotSounding(profileData, sys.argv[2], icao, datetime)
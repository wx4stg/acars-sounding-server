#!/usr/bin/env python3
# Interactive Sounding Plotter
# Created 17 May 2023 by Sam Gardner <sam@wx4stg.com>


import airportsdata
import requests
from metpy.units import units
import metpy.calc as mpcalc
from metpy import plots
from metpy import constants
from metpy.io import add_station_lat_lon, parse_metar_to_dataframe
from ecape.calc import calc_ecape
from os import path
from pathlib import Path
import numpy as np
from datetime import datetime as dt
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeat

import holoviews as hv
import panel as pn

hv.extension('bokeh')


from skew_t_plot import skew_t_plot


def calc_skew_t_offset(pressure, skew_angle):
    pressure_data = pressure.data.to(units.hPa).magnitude
    P_bottom = np.max(pressure_data)
    temp_offset = 37*np.log10(P_bottom/pressure_data)/(np.tan(np.deg2rad(skew_angle))) * units.delta_degC
    temp_offset = xr.DataArray(
        temp_offset,
        dims=("LEVEL_unitless"),
        coords={"LEVEL_unitless": pressure.LEVEL_unitless}
    )
    return temp_offset


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
    soundingDS["skewt_offset"] = calc_skew_t_offset(soundingDS.LEVEL, 30)
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
    soundingDS.attrs["sbLCL"] = mpcalc.lcl(soundingDS.LEVEL[0], soundingDS.TEMP[0], soundingDS.DWPT[0])[0]
    soundingDS.attrs["sbLFC"] = mpcalc.lfc(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_temperature_profile=sbParcelPath)[0]
    soundingDS.attrs["sbEL"] = mpcalc.el(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_temperature_profile=sbParcelPath)[0]
    soundingDS.attrs["sbCAPE"], soundingDS.attrs["sbCINH"] = mpcalc.surface_based_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)

    # most unstable
    initPressure, initTemp, initDewpoint, initIdx = mpcalc.most_unstable_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["mu_initPressure"], soundingDS.attrs["mu_initTemp"], soundingDS.attrs["mu_initDewpoint"] = initPressure, initTemp, initDewpoint
    muParcelPath = xr.full_like(soundingDS.TEMP, np.nan * units.degK)
    muParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], initTemp, initDewpoint)
    soundingDS["muParcelPath"] = muParcelPath
    soundingDS.attrs["muLCL"] = mpcalc.lcl(initPressure, initTemp, initDewpoint)[0]
    soundingDS.attrs["muLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=muParcelPath[initIdx:])[0]
    soundingDS.attrs["muEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=muParcelPath[initIdx:])[0]
    soundingDS.attrs["muCAPE"], soundingDS.attrs["muCINH"] = mpcalc.most_unstable_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    
    # 100-hPa mixed layer
    mlParcelPath = xr.full_like(soundingDS.TEMP, np.nan * units.degK)
    if np.nanmax(soundingDS.LEVEL.data) - np.nanmin(soundingDS.LEVEL.data) > 100 * units.hPa:
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
        initIdx = len(soundingDS.where(soundingDS.LEVEL > initPressure, drop=True).LEVEL.data)
        mlParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], initTemp, initDewpoint)
        soundingDS.attrs["mlLCL"] = mpcalc.lcl(initPressure, initTemp, initDewpoint)[0]
        soundingDS.attrs["mlLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=mlParcelPath[initIdx:])[0]
        soundingDS.attrs["mlEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=mlParcelPath[initIdx:])[0]
        soundingDS.attrs["mlCAPE"], soundingDS.attrs["mlCINH"] = mpcalc.mixed_layer_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    else:
        soundingDS.attrs["mlLCL"] = np.nan * units.hPa
        soundingDS.attrs["mlLFC"] = np.nan * units.hPa
        soundingDS.attrs["mlEL"] = np.nan * units.hPa
        soundingDS.attrs["mlCAPE"] = np.nan * units.joule/units.kilogram
        soundingDS.attrs["mlCINH"] = np.nan * units.joule/units.kilogram
    soundingDS["mlParcelPath"] = mlParcelPath

    # effective inflow layer
    inflowParcelPath = xr.full_like(soundingDS.TEMP, np.nan * units.degK)
    if not np.isnan(inflowBottom):
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_start_pressure=inflowBottom, depth=(inflowBottom - inflowTop))
        initIdx = len(soundingDS.where(soundingDS.LEVEL > initPressure, drop=True).LEVEL.data)
        inflowParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx], soundingDS.DWPT[initIdx])
        soundingDS.attrs["inLCL"] = mpcalc.lcl(initPressure, initTemp, initDewpoint)[0]
        soundingDS.attrs["inLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=inflowParcelPath[initIdx:])[0]
        soundingDS.attrs["inEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=inflowParcelPath[initIdx:])[0]
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
        lowest1500_index = np.argmin(soundingDS.AGL.data <= units.Quantity(1500, 'meter'))
        llj_index = np.argmax(soundingDS.WSPD.data[:lowest1500_index])
        llj_u, llj_v = soundingDS.u[llj_index], soundingDS.v[llj_index]
        soundingDS.attrs["corfidi_up"], soundingDS.attrs["corfidi_down"] = mpcalc.corfidi_storm_motion(soundingDS.LEVEL, soundingDS.u, soundingDS.v, u_llj=llj_u, v_llj=llj_v)
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
    dcape_profile = xr.full_like(soundingDS.TEMP, np.nan * units.degC)
    if np.nanmax(soundingDS.LEVEL.data) > 700 * units.hPa and np.nanmin(soundingDS.LEVEL.data) < 500 * units.hPa:
        dcape_result  = mpcalc.downdraft_cape(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
        dcape_quantity = dcape_result[0]
        dcape_profile[:len(dcape_result[2])] = dcape_result[2]
    else:
        dcape_quantity = np.nan * units.joule/units.kilogram
    soundingDS.attrs["dcape"] = dcape_quantity
    soundingDS["dcape_profile"] = dcape_profile
    soundingDS = soundingDS.drop('index')
    return soundingDS
    

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


def plotSounding(profileData, icao, time, soundingType="Observed"):
    title_text = ''
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
        title_text = f"{soundingType} Sounding -- {time.strftime('%H:%M UTC %d %b %Y')} -- {icao} ({groundLat.magnitude:.2f}, {groundLon.magnitude:.2f})"
    else:
        groundLat = None
        groundLon = None
        title_text = f"{soundingType} Sounding -- {time.strftime('%H:%M UTC %d %b %Y')} -- {icao}"
    tax = pn.pane.Markdown(title_text, styles={'text-align': 'center'})
    skew = skew_t_plot(profileData)
    
    # if groundLat is not None:
    #     thermalWindAx.text(0.5, 0.95, "Thermal Wind\nRel. Humidity", ha="center", va="center", fontsize=9, transform=thermalWindAx.transAxes)
    # else:
    #     thermalWindAx.text(0.5, 0.95, "Rel. Humidity", ha="center", va="center", fontsize=9, transform=thermalWindAx.transAxes)
    # plotThermalWind(profileData, thermalWindAx, groundLat)
    # thermalWindAx.patch.set_alpha(0)
    
    # hodoAx = fig.add_axes([12/20, 9/16, 7/20, 5/16])
    # plotHodograph(profileData, hodoAx)
    # hodoAx.patch.set_alpha(0)
    
    # partialThicknessAx = fig.add_axes([14/20, 7/16, 2/20, 2/16])
    # precipType = plotPartialThickness(profileData, partialThicknessAx)
    
    # psblHazTypeAx = fig.add_axes([12/20, 7/16, 2/20, 2/16])
    # psblHazTypeAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # plotPsblHazType(profileData, psblHazTypeAx, precipType)



    # mapAx = fig.add_axes([16/20, 7/16, 3/20, 2/16], projection=ccrs.PlateCarree())
    # if groundLat is not None and groundLon is not None:
    #     mapAx.scatter(groundLon, groundLat, transform=ccrs.PlateCarree(), color="black", marker="*")
    #     mapAx.add_feature(cfeat.STATES.with_scale("50m"))
    #     mapAx.add_feature(plots.USCOUNTIES.with_scale("5m"), edgecolor="gray", linewidth=0.25)
    # else:
    #     mapAx.text(0.5, 0.5, "Location not available", ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=mapAx.transAxes)
    # mapAx.add_feature(cfeat.COASTLINE.with_scale("50m"))
    
    # thermodynamicsAx = fig.add_axes([1/20, 1/16, 9/20, 3/16])
    # thermodynamicsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # thermodynamicsAx.spines[['right']].set_visible(False)
    # plotThermoynamics(profileData, thermodynamicsAx)
    # thermodynamicsAx.patch.set_alpha(0)

    # paramsAx = fig.add_axes([10/20, 1/16, 2/20, 3/16])
    # paramsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # plotParams(profileData, paramsAx)
    # paramsAx.patch.set_alpha(0)

    # dynamicsAx = fig.add_axes([12/20, 1/16, 7/20, 6/16])
    # dynamicsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    # plotDynamics(profileData, dynamicsAx)

    # px = 1/plt.rcParams["figure.dpi"]
    # fig.set_size_inches(1920*px, 1080*px)
    # width_unit = skew.ax.get_position().width / 10
    # height_unit = skew.ax.get_position().height / 10
    # tax.set_position([1*width_unit, 14*height_unit, 18*width_unit, 1*height_unit])
    # skew.ax.set_position([1*width_unit, 4*height_unit, 10*width_unit, 10*height_unit])
    # thermalWindAx.set_position([11*width_unit, 4*height_unit, width_unit, 10*height_unit])
    # oldHodoLimits = hodoAx.get_xlim(), hodoAx.get_ylim()
    # hodoAx.set_adjustable("datalim")
    # hodoAx.set_position([12*width_unit, 9*height_unit, 7*width_unit, 5*height_unit])
    # hodoaspect = (hodoAx.get_position().height*1080)/(hodoAx.get_position().width*1920)
    # altxmax = ((oldHodoLimits[1][1] - oldHodoLimits[1][0])/hodoaspect)+oldHodoLimits[0][0]
    # altymax = ((oldHodoLimits[0][1] - oldHodoLimits[0][0])*hodoaspect)+oldHodoLimits[1][0]
    # hodoAx.set_xlim(oldHodoLimits[0][0], np.nanmax([altxmax, oldHodoLimits[0][1]]))
    # hodoAx.set_ylim(oldHodoLimits[1][0], np.nanmax([altymax, oldHodoLimits[1][1]]))

    # psblHazTypeAx.set_position([12*width_unit, 7*height_unit, 2*width_unit, 2*height_unit])
    # partialThicknessAx.set_position([14*width_unit, 7*height_unit, 2*width_unit, 2*height_unit])

    # thermodynamicsAx.set_position([1*width_unit, 1*height_unit, 9*width_unit, 3*height_unit])
    # paramsAx.set_position([10*width_unit, 1*height_unit, 2*width_unit, 3*height_unit])
    # dynamicsAx.set_position([12*width_unit, 1*height_unit, 7*width_unit, 6*height_unit])

    # mapAx.set_adjustable("datalim")
    # mapAx.set_position([16*width_unit, 7*height_unit, 3*width_unit, 2*height_unit])
    # Path(path.dirname(outputPath)).mkdir(parents=True, exist_ok=True)
    print(skew)
    fig = pn.Column(tax, pn.Column(pn.Row(skew.skew_t)))
    pn.serve(fig)
    return fig


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: soundingPlot.py <input> <output>")
        exit()
    if not path.exists(sys.argv[1]):
        print("Input file does not exist!")
        exit()
    if sys.argv[1].endswith("acars.nc"):
        readACARS(sys.argv[1])
    else:
        profileData, icao, datetime  = readSharppy(sys.argv[1])
    
    output = plotSounding(profileData, icao, datetime)
    if len(sys.argv) == 3:
        output.save(sys.argv[2])



#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import stdlib modules
import os
import re
import requests
import logging

from pathlib import Path
from urllib.parse import quote_plus, urlparse, parse_qs, unquote_plus

# import external modules
import pandas as pd
import geopandas as gpd
from shapely.wkt import dumps, loads
from shapely.geometry import Polygon, shape
from tqdm import tqdm

# internal OST libs
from ost.helpers.db import pgHandler
from ost.helpers import copernicus as cop

# set up logger
logger = logging.getLogger(__name__)


def _to_shapefile(gdf, outfile, append=False):

    # check if file is there
    if os.path.isfile(outfile):

        # in case we want to append, we load the old one and add the new one
        if append:
            columns = [
                "id",
                "identifier",
                "polarisationmode",
                "orbitdirection",
                "acquisitiondate",
                "relativeorbit",
                "orbitnumber",
                "product_type",
                "slicenumber",
                "size",
                "beginposition",
                "endposition",
                "lastrelativeorbitnumber",
                "lastorbitnumber",
                "uuid",
                "platformidentifier",
                "missiondatatakeid",
                "swathidentifier",
                "ingestiondate",
                "sensoroperationalmode",
                "geometry",
            ]

            # get existing geodataframe from file
            old_df = gpd.read_file(outfile)
            old_df.columns = columns
            # drop id
            old_df.drop("id", axis=1, inplace=True)
            # append new results
            gdf.columns = columns[1:]
            gdf = old_df.append(gdf)

            # remove duplicate entries
            gdf.drop_duplicates(subset="identifier", inplace=True)

        # remove old file
        os.remove(outfile)
        os.remove("{}.cpg".format(outfile[:-4]))
        os.remove("{}.prj".format(outfile[:-4]))
        os.remove("{}.shx".format(outfile[:-4]))
        os.remove("{}.dbf".format(outfile[:-4]))

    # calculate new index
    gdf.insert(loc=0, column="id", value=range(1, 1 + len(gdf)))

    # write to new file
    if len(gdf.index) >= 1:
        gdf.to_file(outfile)
    else:
        logger.info("No scenes found in this AOI during this time")


def _to_geopackage(gdf, outfile, append=False):

    # check if file is there
    if Path(outfile).exists():

        # in case we want to append, we load the old one and add the new one
        if append:
            columns = [
                "id",
                "identifier",
                "polarisationmode",
                "orbitdirection",
                "acquisitiondate",
                "relativeorbit",
                "orbitnumber",
                "product_type",
                "slicenumber",
                "size",
                "beginposition",
                "endposition",
                "lastrelativeorbitnumber",
                "lastorbitnumber",
                "uuid",
                "platformidentifier",
                "missiondatatakeid",
                "swathidentifier",
                "ingestiondate",
                "sensoroperationalmode",
                "geometry",
            ]

            # get existing geodataframe from file
            old_df = gpd.read_file(outfile)
            old_df.columns = columns
            # drop id
            old_df.drop("id", axis=1, inplace=True)
            # append new results
            gdf.columns = columns[1:]
            gdf = old_df.append(gdf)

            # remove duplicate entries
            gdf.drop_duplicates(subset="identifier", inplace=True)

        # remove old file
        Path(outfile).unlink()

    # calculate new index
    gdf.insert(loc=0, column="id", value=range(1, 1 + len(gdf)))

    # write to new file
    if len(gdf.index) > 0:
        gdf.to_file(outfile, driver="GPKG")
    else:
        logger.info("No scenes found in this AOI during this time")


def _to_postgis(gdf, db_connect, outtable):

    # check if tablename already exists
    db_connect.cursor.execute(
        "SELECT EXISTS (SELECT * FROM "
        "information_schema.tables WHERE "
        "LOWER(table_name) = "
        "LOWER('{}'))".format(outtable)
    )
    result = db_connect.cursor.fetchall()
    if result[0][0] is False:
        logger.info(f"Table {outtable} does not exist in the database. Creating it...")
        db_connect.pgCreateS1("{}".format(outtable))
        maxid = 1
    else:
        try:
            maxid = db_connect.pgSQL(f"SELECT max(id) FROM {outtable}")
            maxid = maxid[0][0]
            if maxid is None:
                maxid = 0

            logger.info(
                f"Table {outtable} already exists with {maxid} entries. "
                f"Will add all non-existent results to this table."
            )
            maxid = maxid + 1
        except Exception:
            raise RuntimeError(
                f"Existent table {outtable} does not seem to be compatible " f"with Sentinel-1 data."
            )

    # add an index as first column
    gdf.insert(loc=0, column="id", value=range(maxid, maxid + len(gdf)))
    db_connect.pgSQLnoResp(f"SELECT UpdateGeometrySRID('{outtable.lower()}', 'geometry', 0);")

    # construct the SQL INSERT line
    for _index, row in gdf.iterrows():

        row["geometry"] = dumps(row["footprint"])
        row.drop("footprint", inplace=True)
        identifier = row.identifier
        uuid = row.uuid
        line = tuple(row.tolist())

        # first check if scene is already in the table
        result = db_connect.pgSQL("SELECT uuid FROM {} WHERE " "uuid = '{}'".format(outtable, uuid))
        try:
            result[0][0]
        except IndexError:
            logger.info(f"Inserting scene {identifier} to {outtable}")
            db_connect.pgInsert(outtable, line)
            # apply the dateline correction routine
            db_connect.pgDateline(outtable, uuid)
            maxid += 1
        else:
            logger.info(f"Scene {identifier} already exists within table {outtable}.")

    logger.info(f"Inserted {len(gdf)} entries into {outtable}.")
    logger.info(f"Table {outtable} now contains {maxid - 1} entries.")
    logger.info("Optimising database table.")

    # drop index if existent
    try:
        db_connect.pgSQLnoResp("DROP INDEX {}_gix;".format(outtable.lower()))
    except Exception:
        pass

    # create geometry index and vacuum analyze
    db_connect.pgSQLnoResp("SELECT UpdateGeometrySRID('{}', " "'geometry', 4326);".format(outtable.lower()))
    db_connect.pgSQLnoResp(
        "CREATE INDEX {}_gix ON {} USING GIST " "(geometry);".format(outtable, outtable.lower())
    )
    db_connect.pgSQLnoResp("VACUUM ANALYZE {};".format(outtable.lower()))


def check_availability(inventory_gdf, download_dir, data_mount):
    """This function checks if the data is already downloaded or
       available through a mount point on DIAS cloud

    :param inventory_gdf:
    :param download_dir:
    :param data_mount:
    :return:
    """

    from ost import Sentinel1Scene

    # add download path, or set to None if not found
    inventory_gdf["download_path"] = inventory_gdf.identifier.apply(
        lambda row: str(Sentinel1Scene(row).get_path(download_dir, data_mount))
    )

    return inventory_gdf


def transform_geometry(geometry):

    try:
        geom = Polygon(geometry['coordinates'][0])
    except Exception:
        geom = Polygon(geometry['coordinates'][0][0])

    return geom


def query_dataspace(query, access_token):
    """Query the Copernicus Dataspace OData API.

    Translates the legacy RESTO-style query string (built by Project.py /
    copernicus.py helpers) into the new OData $filter format, since the old
    RESTO endpoint (/resto/api/collections/Sentinel1/search.json) returns 403.
    """

    # ------------------------------------------------------------------
    # 1. Parse the legacy RESTO query string into a dict
    # ------------------------------------------------------------------
    parsed = urlparse(query)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    
    geometry     = params.get('geometry', None)
    start_date   = params.get('startDate', '2014-10-01T00:00:00Z')
    end_date     = params.get('completionDate', '2099-01-01T23:59:59Z')
    product_type = params.get('productType', None)
    sensor_mode  = params.get('sensorMode', None)
    polarisation = params.get('polarisation', None)
    max_records  = int(params.get('maxRecords', 100))

    if polarisation:
        polarisation = unquote_plus(polarisation).replace('&', ' ').strip()

    # ------------------------------------------------------------------
    # 2. Build OData $filter string
    # ------------------------------------------------------------------
    filters = ["Collection/Name eq 'SENTINEL-1'"]

    start_odata = start_date.replace('T00:00:00Z', 'T00:00:00.000Z')
    end_odata   = end_date.replace('T23:59:59Z', 'T23:59:59.000Z')
    filters.append("ContentDate/Start gt " + start_odata)
    filters.append("ContentDate/Start lt " + end_odata)

    if product_type:
        filters.append(
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            "and att/OData.CSC.StringAttribute/Value eq '" + product_type + "')"
        )

    if sensor_mode:
        filters.append(
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' "
            "and att/OData.CSC.StringAttribute/Value eq '" + sensor_mode + "')"
        )

    if geometry:
        geom_wkt = unquote_plus(geometry).strip()
        filters.append("OData.CSC.Intersects(area=geography'SRID=4326;" + geom_wkt + "')")

    filter_str = ' and '.join(filters)

    # ------------------------------------------------------------------
    # 3. Page through OData results
    # ------------------------------------------------------------------
    odata_base = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
    headers    = {'Authorization': 'Bearer ' + access_token}

    logger.info('Querying the Copernicus Dataspace OData Server')
    logger.info('Filter: ' + filter_str)

    dfs, skip, top, total = [], 0, min(max_records, 100), 0

    while True:
        odata_params = {
            '$filter': filter_str,
            '$top':    top,
            '$skip':   skip,
            '$expand': 'Attributes',
        }
        try:
            response = requests.get(
                odata_base,
                headers=headers,
                params=odata_params,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error('OData API request failed: ' + str(e))
            break

        features = data.get('value', [])
        if not features:
            break

        dfs.append(pd.DataFrame(features))
        total += len(features)
        logger.info('Retrieved ' + str(total) + ' products so far...')

        if '@odata.nextLink' in data and total < max_records:
            skip += top
        else:
            break

    if not dfs:
        raise ValueError('No products found for the given search parameters.')

    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise ValueError('No products found for the given search parameters.')

    logger.info('Found ' + str(len(df)) + ' total products')
    tqdm.pandas()

    # ------------------------------------------------------------------
    # 4. Helper functions to extract OData attributes
    # ------------------------------------------------------------------
    def _extract_attr(attrs, name):
        if not isinstance(attrs, list):
            return None
        for a in attrs:
            if a.get('Name') == name:
                return a.get('Value')
        return None

    def _parse_row(row):
        attrs = row.get('Attributes', [])
        name  = row.get('Name', '')

        identifier        = name[:-5] if name.endswith('.SAFE') else name
        # OData uses 'polarisationChannels' with value e.g. 'VV&VH'
        polarisationmode  = (_extract_attr(attrs, 'polarisationChannels') or '').replace('&', ' ')
        orbitdirection    = _extract_attr(attrs, 'orbitDirection') or ''
        beginposition     = (row.get('ContentDate') or {}).get('Start', '')
        endposition       = (row.get('ContentDate') or {}).get('End', '')
        acquisitiondate   = beginposition[:10]
        relativeorbit     = _extract_attr(attrs, 'relativeOrbitNumber') or ''
        orbitnumber       = _extract_attr(attrs, 'absoluteOrbitNumber') or ''
        producttype       = _extract_attr(attrs, 'productType') or ''
        slicenumber       = _extract_attr(attrs, 'sliceNumber') or ''
        missiondatatakeid = _extract_attr(attrs, 'missionDatatakeId') or ''
        sensormode        = _extract_attr(attrs, 'operationalMode') or ''
        size              = str(row.get('ContentLength', ''))
        uuid              = row.get('Id', '')
        platformidentifier = _extract_attr(attrs, 'platformShortName') or ''
        swathidentifier   = _extract_attr(attrs, 'swathIdentifier') or ''
        ingestiondate     = row.get('PublicationDate', '')
        lastrelativeorbit = _extract_attr(attrs, 'lastRelativeOrbitNumber') or relativeorbit
        lastorbitnumber   = _extract_attr(attrs, 'lastOrbitNumber') or orbitnumber

        footprint = row.get('Footprint', '')
        geom = None
        if footprint and isinstance(footprint, str):
            wkt_match = re.search(r'POLYGON[\s\S]*', footprint)
            if wkt_match:
                try:
                    geom = loads(wkt_match.group(0).rstrip("'"))
                except Exception:
                    geom = None

        return pd.Series({
            'identifier':              identifier,
            'polarisationmode':        polarisationmode,
            'orbitdirection':          orbitdirection,
            'acquisitiondate':         acquisitiondate,
            'relativeorbit':           relativeorbit,
            'orbitnumber':             orbitnumber,
            'producttype':             producttype,
            'slicenumber':             slicenumber,
            'size':                    size,
            'beginposition':           beginposition,
            'endposition':             endposition,
            'lastrelativeorbitnumber': lastrelativeorbit,
            'lastorbitnumber':         lastorbitnumber,
            'uuid':                    uuid,
            'platformidentifier':      platformidentifier,
            'missiondatatakeid':       missiondatatakeid,
            'swathidentifier':         swathidentifier,
            'ingestiondate':           ingestiondate,
            'sensoroperationalmode':   sensormode,
            'geometry':                geom,
        })

    logger.info('Normalising OData response to legacy OST schema...')
    gdf = gpd.GeoDataFrame(
        df.progress_apply(_parse_row, axis=1),
        geometry='geometry',
        crs='epsg:4326'
    )

    # filter by polarisation if specified
    # OData returns 'VV&VH', normalise both sides by stripping spaces and &
    if polarisation:
        pol_filter = polarisation.replace(' ', '').replace('&', '')
        gdf = gdf[
            gdf['polarisationmode']
            .str.replace(' ', '', regex=False)
            .str.replace('&', '', regex=False)
            .str.contains(pol_filter, na=False)
        ]
        logger.info('After polarisation filter (' + polarisation + '): ' + str(len(gdf)) + ' products')

    if gdf.empty:
        raise ValueError('No products found for the given search parameters.')

    return gdf


def dataspace_catalogue(
    query_string,
    output,
    append=False,
    uname=None,
    pword=None,
    base_url="https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel1/search.json?",
):
    """This is the main search function on Copernicus Dataspace.

    :param query_string:
    :param output:
    :param append:
    :param uname:
    :param pword:
    :param base_url:
    :return:
    """

    # retranslate Path object to string
    output = str(output)

    # get connected to dataspace
    access_token = cop.get_access_token(uname, pword)
    query = base_url + query_string
    query = quote_plus(query, safe=":/?&=%")
    logger.info(f"FULL_QUERY_URL: {query}")

    # get the catalogue in a dict
    gdf = query_dataspace(query, access_token)

    if output[-4:] == ".shp":
        logger.info(f"Writing inventory data to shape file: {output}")
        _to_shapefile(gdf, output, append)
    elif output[-5:] == ".gpkg":
        logger.info(f"Writing inventory data to geopackage file: {output}")
        _to_geopackage(gdf, output, append)
    else:
        logger.info(f"Writing inventory data to PostGIS table: {output}")
        db_connect = pgHandler()
        _to_postgis(gdf, db_connect, output)
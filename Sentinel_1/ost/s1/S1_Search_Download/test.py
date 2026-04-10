from ost.s1.search_data import dataspace_catalogue
from ost.helpers import copernicus as cop
from pathlib import Path

query = "geometry=POLYGON((...))"  # dein AOI
output = Path("test_inventory.gpkg")

dataspace_catalogue(
    query_string=query,
    output=output,
    uname="jowi1438@student.su.se",
    pword=input("Password: ")
)
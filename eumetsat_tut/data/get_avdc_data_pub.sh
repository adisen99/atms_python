#!/bin/bash
#---------------------------------------------------------------------------------------------------
#
# name:			get_avdc_data_pub.sh
#
#---------------------------------------------------------------------------------------------------

echo 
echo "=================================================================================================================="
echo 
echo "Get AVDC data from public directories"
echo 
echo "=================================================================================================================="
echo 

if [ $# -ne 1 ]; then
	echo
	echo "usage: get_avdc_data_pub.sh [AVDC web dir]"
	echo
	echo "------------------------------------------------------------------------------------------"
	echo "1: AVDC web dir: e.g. https://avdc.gsfc.nasa.gov/pub/data/satellite/Aura/OMI/V03/L3/OMNO2d_HR/"
	echo "------------------------------------------------------------------------------------------"
	exit
fi

DIRHTTP=$1

wget -r -m -e robots=off -nH --no-parent --cut-dirs=9 --reject "*.html*" $DIRHTTP

import requests

# apikey = 'AzqwaSy00000biJUpolkhhagg2U4Ynhop0mIg'
apikey = "your_api_key"


def get_altitude(lat, lon):
    url = "https://maps.googleapis.com/maps/api/elevation/json?locations={},{}&key={}".format(
        lat, lon, apikey
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            altitude = data["results"][0]["elevation"]
            return altitude
    return None


# print(get_altitude(33.522506375758200000,-5.358603000640869))

sites = {
    "Casablanca": [33.368444860787600000, -7.577004432678220000],
    "Fes": [33.930606750397100000, -4.974113702774050000],
    "Tetouan": [35.485694000000000000, -5.359000000000000000],
    "Tit_Melil": [33.591169058875300000, -7.455790042877200000],
    "Marrakech": [31.607014047858700000, -8.051170706748960000],
    "OuladYahia": [33.522506375758200000, -7.248916625976560000],
}


def get_altitude_context():
    return {
        "Casablanca": 196.7499695,
        "Fes": 571.6484375,
        "Tetouan": 1206.678223,
        "Tit Melil": 97.62522888,
        "Marrakech": 458.2503967,
    }


# # Get altitude for existing elevation
# context_altitude = get_altitude_context()
# el_Casablanca = context_altitude['Casablanca']
# el_Fes = context_altitude['Fes']
# el_Tetouan = context_altitude['Tetouan']
# el_Tit_Melil = context_altitude['Tit Melil']
# el_Marrakech = context_altitude['Marrakech']

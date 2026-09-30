import requests


def get_weather(city: str) -> str:
    """Get the current temperature for a specified city."""

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1
        }
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Data for {city} was not found."

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m"
        }
    )

    weather_data = weather_response.json()

    temperature = weather_data["current"]["temperature_2m"]

    return f"The current temperature in {city} is {temperature}°C."


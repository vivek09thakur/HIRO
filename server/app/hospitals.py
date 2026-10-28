"""Nearby hospital lookup via OpenStreetMap (Overpass mirrors, Nominatim as fallback)."""

import asyncio
import math
import ssl
import time

import httpx
import truststore

OVERPASS_URLS = (
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "HIRO symptom checker (educational, local)"
REQUEST_TIMEOUT = 12.0
CACHE_TTL_SECONDS = 600
CACHE_MAX_ENTRIES = 200
# Use the OS certificate store so TLS works behind corporate proxies that re-sign traffic.
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

_cache: dict[tuple[float, float, int], tuple[float, list[dict]]] = {}


class LookupUnavailable(Exception):
    """Every upstream source failed."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


async def _query_mirror(client: httpx.AsyncClient, url: str, query: str) -> list[dict]:
    response = await client.post(url, data={"data": query})
    response.raise_for_status()
    body = response.json()
    # Overpass reports server-side timeouts as HTTP 200 with a remark and no elements.
    if not body.get("elements") and "remark" in body:
        raise httpx.HTTPError(f"{url}: {body['remark']}")
    return body.get("elements", [])


async def _fetch_elements(client: httpx.AsyncClient, query: str) -> list[dict]:
    """Race the mirrors; return the first non-empty answer, or [] once all have answered empty."""
    tasks = [asyncio.create_task(_query_mirror(client, url, query)) for url in OVERPASS_URLS]
    last_error: Exception = httpx.HTTPError("Overpass unavailable")
    saw_empty = False
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    elements = task.result()
                except (httpx.HTTPError, ValueError) as exc:
                    last_error = exc
                    continue
                if elements:
                    return elements
                saw_empty = True
    finally:
        for task in tasks:
            task.cancel()
    if saw_empty:
        return []
    raise last_error


async def _fetch_nominatim(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int) -> list[dict]:
    """Fallback: bounded Nominatim searches for hospitals and clinics, mapped to Overpass-like elements."""
    dlat = radius_m / 111_000
    dlon = radius_m / (111_000 * max(math.cos(math.radians(lat)), 0.2))
    viewbox = f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}"
    elements: list[dict] = []
    for amenity in ("hospital", "clinic"):
        response = await client.get(
            NOMINATIM_URL,
            params={"q": amenity, "format": "jsonv2", "limit": 25, "bounded": 1, "viewbox": viewbox},
        )
        response.raise_for_status()
        for place in response.json():
            elements.append(
                {
                    "lat": float(place["lat"]),
                    "lon": float(place["lon"]),
                    "tags": {
                        "name": place.get("name") or place.get("display_name", "").split(",")[0],
                        "amenity": place.get("type") if place.get("type") in ("hospital", "clinic") else amenity,
                        "addr:city": place.get("address", {}).get("city"),
                    },
                }
            )
    return elements


async def nearby_hospitals(lat: float, lon: float, radius_m: int, limit: int = 20) -> list[dict]:
    # ~100 m grid so nearby repeat requests share one Overpass call.
    cache_key = (round(lat, 3), round(lon, 3), radius_m)
    now = time.monotonic()
    hit = _cache.get(cache_key)
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]

    query = (
        f'[out:json][timeout:10];'
        f'nwr["amenity"~"^(hospital|clinic)$"](around:{radius_m},{lat},{lon});'
        f'out center tags;'
    )
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}, verify=SSL_CONTEXT) as client:
        try:
            elements = await _fetch_elements(client, query)
        except (httpx.HTTPError, ValueError) as overpass_error:
            try:
                elements = await _fetch_nominatim(client, lat, lon, radius_m)
            except (httpx.HTTPError, ValueError, KeyError) as nominatim_error:
                raise LookupUnavailable(
                    f"Overpass: {type(overpass_error).__name__}; Nominatim: {type(nominatim_error).__name__}"
                ) from nominatim_error

    seen: set[str] = set()
    results: list[dict] = []
    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name")
        center = element.get("center", {})
        plat = element.get("lat", center.get("lat"))
        plon = element.get("lon", center.get("lon"))
        if not name or plat is None or plon is None or name in seen:
            continue
        seen.add(name)
        address = ", ".join(
            part for part in (tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")) if part
        )
        results.append(
            {
                "name": name,
                "kind": tags.get("amenity", "hospital"),
                "distance_km": round(haversine_km(lat, lon, plat, plon), 1),
                "address": address or None,
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "emergency": tags.get("emergency") == "yes",
                "lat": plat,
                "lon": plon,
                "directions_url": f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}",
            }
        )

    results.sort(key=lambda h: (h["kind"] != "hospital", h["distance_km"]))
    results = results[:limit]

    if len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))
    _cache[cache_key] = (now, results)
    return results

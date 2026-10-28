import asyncio
import time

from server.app.hospitals import nearby_hospitals

t = time.perf_counter()
try:
    data = asyncio.run(nearby_hospitals(28.6139, 77.2090, 3000))
    print(f"OK {len(data)} results in {time.perf_counter()-t:.1f}s")
    for h in data[:3]:
        print(" ", h["name"], "|", h["kind"], "|", h["distance_km"], "km |", h["address"])
except Exception as exc:
    print(f"EXC {type(exc).__name__} {exc!r} after {time.perf_counter()-t:.1f}s")

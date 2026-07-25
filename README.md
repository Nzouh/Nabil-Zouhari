# zouhari.dev

Personal site of Nabil Zouhari: portfolio, articles, and the code behind them.
Deployed on Vercel as a static site (no build step).

## Structure

```
frontend/                        # everything Vercel serves
  index.html                     # portfolio homepage        -> /
  articles.html                  # article listing           -> /articles
  articles/geomstats-lab/        # article + figures + notebook
    index.html                   #                           -> /articles/weather-on-a-sphere
backend/                         # research code behind the articles
  geomstats-lab/                 # scripts, data caches, article draft
vercel.json                      # static config + rewrites
```

## Articles

**Weather Data Lives on a Sphere. Our Statistics Should Too.**
Geometric statistics for weather data on Earth, built with
[GeomStats](https://geomstats.github.io/) and [ERA5](https://cds.climate.copernicus.eu/):
Frechet means, geodesic distances, and Principal Geodesic Analysis on the sphere, applied to
real temperature fields, with the Euclidean failure modes measured rather than assumed.

- Read: [frontend/articles/geomstats-lab/index.html](frontend/articles/geomstats-lab/index.html)
  (or `/articles/weather-on-a-sphere` on the deployed site)
- Reproduce: [companion notebook](frontend/articles/geomstats-lab/companion_notebook.html)
  runs the whole evidence base end to end

Methods are Karcher (1977) and Fletcher et al. (2004), as surveyed in Papillon, Sanborn,
Mathe et al., *Beyond Euclid* (Mach. Learn.: Sci. Technol. 6 031002, 2025;
[arXiv:2407.09468](https://arxiv.org/abs/2407.09468)).

## Running the geomstats lab

Python **3.12** (geomstats 2.8.0 needs `numpy<2.1`, which has no Python 3.13 wheels):

```
cd backend/geomstats-lab
pip install -r requirements.txt

python tool1_centroid_tracker.py        # Frechet-mean benchmarks (B1-B6)
python tool3_geodesic_pca.py            # PGA benchmarks (B1-B6)
python spread_gap_figure.py             # geometry figure
python equivariance_figure.py           # group-theory figure
python era5_heatwave.py                 # heat-dome track + figure
python era5_pga_interannual.py          # 30-year PGA + figure
```

Figures are written to the working directory; the copies the article serves live in
`frontend/articles/geomstats-lab/`, so copy regenerated figures there.

The two ERA5 scripts and the notebook run in **quick mode** against the cached netCDF files
in `backend/geomstats-lab/`, so no API key is needed. Delete the caches to re-download from
the Copernicus Climate Data Store, which requires a free CDS account, a Personal Access Token
in `~/.cdsapirc`, and accepting the ERA5 licence on the dataset page.

## Data attribution

Contains modified Copernicus Climate Change Service information (ERA5). Neither the European
Commission nor ECMWF is responsible for any use of this information. Coastlines from
[Natural Earth](https://www.naturalearthdata.com/) (public domain).

## Author

Nabil Zouhari, 2026.

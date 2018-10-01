# High-Altitude Long-Endurance Aircraft Trajectory Optimization

This repository contains the code used to generate the results in the paper "Dynamic Optimization of High-Altitude Solar Aircraft Trajectories Under Station-Keeping Constraints".  Results from the paper can also be viewed on an [online dashboard](http://dashboard.apmonitor.com/).

The code produces energy optimal trajectories for solar powered HALE aircraft using the [GEKKO Dynamic Optimization Suite](https://gekko.readthedocs.io/en/latest/#).

![Optimized Path](./Images/dashboard.png?raw=true "Optimized Flight Path Results")

## Getting Started

### Prerequisites

The project requires that the following Python libraries be installed in addition to the Anaconda base set.

```
pip install gekko jsonpickle
```
If you would like to use the results viewing dashboard locally, you will also need to install [Dash](https://plot.ly/products/dash/).

```
pip install dash==0.27.0  # The core dash backend
pip install dash-html-components==0.13.2  # HTML components
pip install dash-core-components==0.30.2  # Supercharged components
```

Solar and atmospheric data is calculated using the SMARTS package.  Several example sets of data are included, covering the winter solstice, spring equinox, summer solstice, and fall equinox at 60,000 ft above Albuquerque New Mexico.  If you would like to optimize a trajectory for a different location or day, you will need to download the [SMARTS executable from NREL.](https://www.nrel.gov/grid/solar-resource/smarts.html)  Place the downloaded executable in the PySMARTS folder.

## Running the code

Running main.py with the default settings will produce the winter solstice optimization case discussed in the paper.  A new time-stamped folder will be generated inside of the Results folder.  First a steady state circular orbit will be integrated, and then the optimization will be performed.  By default the optimization will be solved on the BYU public GEKKO server, and should take 10-15 hours depending on server load.  The optimized trajectory data will be written to a file called opt_results.xlsx, while the steady state circular orbit will be under ss_results.xlsx.  

### Settings

Settings for the optimization, aircraft and mission are stored in a settings object.  To change settings, either edit the class file settings.py or change the attributes of the instantiated config object. 

### Viewing Dashboard

To use the browser-based viewing dashboard, run the dashboard.py file after starting an optimization.  Refreshing the page will update the displayed results. 

## Custom Aircraft

The default aircraft used in the optimization is a large flying wing.  To use another aircraft, create a new class in the Aircraft folder that inherits the aircraft_template class.  Then override the necessary parameters and methods.  See the wing_e216.py file for an example.

## Custom Date/Location

To create a mission with a custom date or location, edit the SolarLocations class in solar.py

## Built With

* [GEKKO](https://gekko.readthedocs.io/en/latest/index.html) - The dynamic optimization package used to calculate the optimized trajectories.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

import numpy as np
from copy import deepcopy
from typing import Dict
from sinergym.config.modeling import ModelJSON


class CustomModelJSON(ModelJSON):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def apply_weather_variability(
            self,
            variation: Dict[str, np.ndarray]) -> str:
        """Modify weather data using Ornstein-Uhlenbeck process for multiple variables.

        Args:
            variation (Dict[str, np.ndarray]): Dictionary where keys are column names and values are 
                                                numpy arrays with the sigma, mean, and tau for the OU process.

        Returns:
            str: New EPW file path generated in simulator working path in that episode or current EPW path if variation is not defined.
        """
        # Deep copy for weather_data
        weather_data_mod = deepcopy(self.weather_data)
        filename = self._weather_path.split('/')[-1]

        # Apply variation to EPW if exists
        if variation:
            # Get dataframe with weather series
            df = weather_data_mod.get_weather_series()

            for column, params in variation.items():
                sigma, mu, tau = params  # Unpack parameters for each variable
                
                T = 1.  # Total time.
                n = len(df[column])  # Number of rows for the column
                dt = T / n
                sigma_bis = sigma * np.sqrt(2. / tau)
                sqrtdt = np.sqrt(dt)

                x = np.zeros(n)

                # Create noise for each time step
                for i in range(n - 1):
                    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                            sigma_bis * sqrtdt * np.random.randn()

                # Add noise to the column data
                df[column] += x

            # Save modified weather data
            weather_data_mod.set_weather_series(df)

            # Change filename to specify variation nature
            filename = filename.split('.epw')[0] + '_Randomized.epw'

            self.logger.debug('Variation applied.')

        episode_weather_path = self.episode_path + '/' + filename
        weather_data_mod.to_epw(episode_weather_path)

        self.logger.debug(
            'Saving episode weather path... [{}]'.format(episode_weather_path))

        return episode_weather_path

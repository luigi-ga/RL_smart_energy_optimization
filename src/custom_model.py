import os
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
            variation (Dict[str, np.ndarray]): Maps columns to be affected to the corresponding Ornstein-Uhlenbeck process parameters.
                The OU parameters should be specified as a Tuple with the sigma, mean and tau for the OU process.
                For example, one could pass {'drybulb': (1, 0, 0.001)} to make the drybulb temperatures change according to an OU with
                parameters 1, 0, and 0.001 and sigma, mean, and tau respectively.
        Returns:
            str: New EPW file path generated in simulator working path in that episode or current EPW path if variation is not defined.
        """

        # Update weather variability
        self.weather_variability = variation

        # Deep copy for weather_data
        weather_data_mod = deepcopy(self.weather_data)

        # Apply variation to EPW if exists
        if variation:
            # Get dataframe with weather series
            df = weather_data_mod.get_weather_series()

            # Iterate over each column and apply variation
            for column, params in variation.items():
                sigma, mu, tau = params  # Unpack parameters for each variable
                
                T = 1.  # Total time.
                n = len(df[column])  # Number of rows for the column
                dt = T / n
                sigma_bis = sigma * np.sqrt(2. / tau)
                sqrtdt = np.sqrt(dt)

                x = np.zeros(n)

                # TODO: include weather_bounds and clip according to them

                # Create noise for each time step
                for i in range(n - 1):
                    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                            sigma_bis * sqrtdt * np.random.randn()

                # Add noise to the column data
                df[column] += x

            # Save modified weather data
            weather_data_mod.set_weather_series(df)

            # Change filename to specify variation nature
            filename = self._weather_path.split('/')[-1]
            filename = filename.split('.epw')[0] + '_Random_%s_%s_%s.epw' % (str(sigma), str(mu), str(tau))

            self.logger.debug('Variation applied.')

        # Save modified weather data to EPW file
        episode_weather_path = self.episode_path + '/' + filename
        weather_data_mod.to_epw(episode_weather_path)

        self.logger.debug('Saving episode weather path... [{}]'.format(episode_weather_path))

        return episode_weather_path
    
    def _check_eplus_config(self) -> None:
        """Check Eplus Environment config definition is correct.
        """

        # COMMON
        # Check weather files exist
        for w_file in self.weather_files:
            w_path = os.path.join(
                self.pkg_data_path, 'weather', w_file)
            try:
                assert os.path.isfile(w_path)
            except AssertionError as err:
                self.logger.critical(
                    'Weather files: {} is not a weather file available in Sinergym.'.format(w_file))
                raise err

        # EXTRA CONFIG
        if self.config is not None:
            for config_key in self.config.keys():
                # Check config parameters values
                # Timesteps
                if config_key == 'timesteps_per_hour':
                    try:
                        assert self.config[config_key] > 0
                    except AssertionError as err:
                        self.logger.critical(
                            'Extra Config: timestep_per_hour must be a positive int value.')
                        raise err
                # Runperiod
                elif config_key == 'runperiod':
                    try:
                        assert isinstance(
                            self.config[config_key], tuple) and len(
                            self.config[config_key]) == 6
                    except AssertionError as err:
                        self.logger.critical(
                            'Extra Config: Runperiod specified in extra configuration has an incorrect format (tuple with 6 elements).')
                        raise err
                # NEW: Weather bounds
                elif config_key == 'weather_bounds':
                    try:
                        assert isinstance(
                            self.config[config_key], dict)
                    except AssertionError as err:
                        self.logger.critical(
                            'Extra Config: Weather bounds specified in extra configuration has an incorrect format (dict).')
                        raise err
                # NEW: Random month
                elif config_key == 'random_month':
                    try:
                        assert isinstance(
                            self.config[config_key], bool)
                    except AssertionError as err:
                        self.logger.critical(
                            'Extra Config: Random month specified in extra configuration has an incorrect format (bool).')
                        raise err
                else:
                    self.logger.error(
                        'Extra Config: Key name specified in config called [{}] is not available in Sinergym.'.format(config_key))
                    
        # TODO: add action_definition if needed




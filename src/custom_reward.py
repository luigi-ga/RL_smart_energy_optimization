"""Implementation of custom reward"""

from typing import Dict, List, Tuple, Union, Any
from sinergym.utils.rewards import LinearReward

class FangerReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        ppd_variable: Union[str, list],
        occupancy_variable: Union[str, list],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,                         # ρ in the paper
        lambda_energy: float = 1e-4,                        # λ_E in the paper
        lambda_ppd: float = 0.1,                            # λ_P in the paper
        lambda_temperature: float = 1.0
    ):
        """
        Linear reward function using Fanger PPD thermal comfort.

        It considers the energy consumption and the PPD thermal comfort metric, as well as occupancy.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * Fanger_metric * (occupancy > 0)

        Args:       
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            ppd_variable (Union[str, list]): Name(s) of the PPD variable(s).
            occupancy_variable (Union[str, list]): Name(s) of the occupancy variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_ppd(float, optional): Constant for removing dimensions from ppd. Defaults to 0.1.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        # Call the parent constructor
        super(FangerReward, self).__init__(
            temperature_variables=temperature_variables,
            energy_variables=energy_variables,
            range_comfort_winter=range_comfort_winter,
            range_comfort_summer=range_comfort_summer,
            summer_start=summer_start,
            summer_final=summer_final,
            energy_weight=energy_weight,
            lambda_energy=lambda_energy,
            lambda_temperature=lambda_temperature
        )

        # Name of the variables
        self.ppd_name = ppd_variable
        self.occupancy_name = occupancy_variable

        # Reward parameters
        self.lambda_temp = lambda_ppd

    def _get_comfort(self,
                     obs_dict: Dict[str,
                                    Any]) -> Tuple[float,
                                                   List[float]]:
        """Calculate the comfort term of the reward.
        Returns:
            Tuple[float, List[float]]: comfort penalty and List with temperatures used.
        """

        # Get the PPD and occupancy values
        ppds = [v for k, v in obs_dict.items() if k in self.ppd_name]
        occupancies = [v for k, v in obs_dict.items() if k in self.occupancy_name]

        # Initialize comfort
        comfort = 0.0

        # Iterate over each zone
        for ppd, occupancy in zip(ppds, occupancies):
            # Calculate the comfort for each zone
            zone_comfort = ppd * (occupancy > 0)
            # If ppd < 20% it is within ASHRAE standards so it is not penalized
            if zone_comfort >= 20:
                comfort += zone_comfort

        # Extract the temperature values
        temp_values = [v for k, v in obs_dict.items() if k in self.temp_names]
        
        # Return the comfort and the temperature
        return comfort, temp_values
import time
import numpy as np
import pandas as pd
import os
import json
import gym
import torch
import init_info
from gym import spaces
from dispatch1 import lambda_iteration1
from dispatch_ipopt_new import ipopt_solve
from datetime import datetime

                                          
                                        
y_54unit = 'data/y_unit54.csv'
busload = 'data/busloads_118_IES.csv'
ParameterData_train_demand = 'data/train_data_54gen.csv'          
Gen_units_path = 'data/kazarlis_units_54.csv'        
CHP_units_path = 'data/kazarlis_units_IES_CHP.csv'
HeatEB_units_path = 'data/kazarlis_units_IES_HeatEB.csv'        
DEFAULT_NUM_GEN = 54
DEFAULT_DISPATCH_FREQ_MINS = 5
DEFAULT_EPISODE_LENGTH_HRS = 24
average_demand = pd.read_csv(os.path.join(ParameterData_train_demand))["demand"].mean()


class Env(object):
    def __init__(self, gen_info, CHP_info, HeatEB_info, profiles_df, busload_df,
                 mode='train', **kwargs):
        self.mode = mode                                                                   
        self.gen_info = gen_info        
        self.CHP_info = CHP_info
        self.HeatEB_info = HeatEB_info
        self.profiles_df = profiles_df        
        self.busload_df = busload_df
        self.dispatch_freq_mins = kwargs.get('dispatch_freq_mins',
                                             DEFAULT_DISPATCH_FREQ_MINS)                                 
        self.dispatch_resolution = self.dispatch_freq_mins / 60.
        self.num_gen = self.gen_info.shape[0]         
        self.num_chp = self.CHP_info.shape[0]         
        self.num_heateb = self.HeatEB_info.shape[0]         
        if self.mode == 'test':
            self.episode_length = len(self.profiles_df)
        else:
            self.episode_length = kwargs.get('episode_length_hrs', DEFAULT_EPISODE_LENGTH_HRS)
            self.episode_length = int(self.episode_length * (60 / self.dispatch_freq_mins))
                        
        self.max_output = self.gen_info['max_output'].to_numpy()              
        self.min_output = self.gen_info['min_output'].to_numpy()              
        self.p_max_vec = self.max_output
        self.p_min_vec = self.min_output
        self.status = self.gen_info['status'].to_numpy()            
                                                            
        self.a = self.gen_info['a'].to_numpy()
        self.b = self.gen_info['b'].to_numpy()
        self.c = self.gen_info['c'].to_numpy()               
        self.t_min_down = self.gen_info['t_min_down'].to_numpy()                                      
        self.t_min_up = self.gen_info['t_min_up'].to_numpy()                                    
        self.t_max_up = self.gen_info['t_max_up'].to_numpy()                                    
                                                              
        self.RampUp = self.gen_info['RampUp'].to_numpy()
        self.RampDown = self.gen_info['RampDown'].to_numpy()
                                                         
        self.min_demand = np.max(self.min_output)
        self.max_demand = np.sum(self.max_output)
                                                      
        self.dispatch_tolerance = 1                                 
        self.day_cost = 0                           
        self.Min_output_CHP = self.CHP_info['MinRunCapacity'].to_numpy()
        self.Max_output_CHP = self.CHP_info['MaxRunCapacity'].to_numpy()
        self.Min_output_CHP_vec = self.Min_output_CHP
        self.Max_output_CHP_vec = self.Max_output_CHP
        self.CHP_status = self.CHP_info['status'].to_numpy()            
        self.t_min_down_CHP = self.CHP_info['DT'].to_numpy()                                      
        self.t_min_up_CHP = self.CHP_info['UT'].to_numpy()                                    
        self.HeatEfficiency = self.CHP_info['HeatEfficiency'].to_numpy()
        self.ElectricalEfficiency = self.CHP_info['ElectricalEfficiency'].to_numpy()
        self.RampUp_CHP = self.CHP_info['RampUp'].to_numpy()
        self.RampDown_CHP = self.CHP_info['RampDown'].to_numpy()
        self.operationalcost = self.HeatEB_info['operationalcost'].to_numpy()
        self.min_output_Heateb = self.HeatEB_info['min_output'].to_numpy()
        self.max_output_Heateb = self.HeatEB_info['max_output'].to_numpy()
        self.min_output_Heateb_vec = self.min_output_Heateb
        self.max_output_Heateb_vec = self.max_output_Heateb
        self.HeatEB_status = self.HeatEB_info['status'].to_numpy()
        self.t_min_up_HeatEB = self.HeatEB_info['t_min_up'].to_numpy()
        self.t_min_down_HeatEB = self.HeatEB_info['t_min_down'].to_numpy()
                  
        self.summarize_marginal_functions()
                                      
                                                                       
                                                                      
                                    

                                  
    def _get_state(self):
        state = {'status': self.status,
                 'CHP_status': self.CHP_status,
                 'HeatEB_status': self.HeatEB_status,
                 'demand_forecast': self.episode_forecast,
                 'demand_heat': self.episode_heat,
                 'demand_gas': self.episode_gas,
                 'solar_forecast': self.episode_solar_forecast,
                 'cost': self.fuel_cost,
                 'timestep': self.episode_timestep,
                 'power': self.power,
                 'power_c': self.power_c,
                 'power_e': self.power_e,
                 'day_cost': self.day_cost,
                 'day': self.day,
                 'ens': self.ens_amount,
                 'cons': self.cons,
                 'load': self.load
                 }
        self.state = state
        return state

    def get_current_state(self):
                             
                                                  
                             
                                                    
                    
        self.determine_priority_orders()
                        
        self.identify_must_ON_and_must_OFF_units()
                                              
        state = self._get_state()
        return state

    def get_next_state(self, action_coal_vec, action_chp_vec, action_eb_vec):
                   
        self.update_gen_status(action_coal_vec, action_chp_vec, action_eb_vec)
                                                     
                                  
        self._update_production_capacities(action_coal_vec, action_chp_vec, self.power, self.power_c)
                 
        self.commits_vec = action_coal_vec
                
        next_state_dict = self.get_current_state()
        return next_state_dict

                                     
    def _determine_constraints(self):
                   
        self.must_on = np.array(
            [True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array(
            [True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])
                  
        self.must_on_chp = np.array(
            [True if 0 < self.CHP_status[i] < self.t_min_up_CHP[i] else False for i in range(self.num_chp)])
        self.must_off_chp = np.array(
            [True if -self.t_min_down_CHP[i] < self.CHP_status[i] < 0 else False for i in range(self.num_chp)])
                     
        self.must_on_heateb = np.array(
            [True if 0 < self.HeatEB_status[i] < self.t_min_up_HeatEB[i] else False for i in range(self.num_heateb)])
        self.must_off_heateb = np.array(
            [True if -self.t_min_down_HeatEB[i] < self.HeatEB_status[i] < 0 else False for i in range(self.num_heateb)])

                                                                                                    
    def _legalise_action_coal(self, action):
        x = np.logical_or(np.array(action), self.must_on)
        x = x * np.logical_not(self.must_off)
        return (np.array(x, dtype=int))

    def _legalise_action_chp(self, action):
        x = np.logical_or(np.array(action), self.must_on_chp)
        x = x * np.logical_not(self.must_off_chp)
        return (np.array(x, dtype=int))

    def _legalise_action_eb(self, action):
        x = np.logical_or(np.array(action), self.must_on_heateb)
        x = x * np.logical_not(self.must_off_heateb)
        return (np.array(x, dtype=int))

                                                
    def _is_legal(self, action):
        action = np.array(action)
        illegal_on = np.any(action[self.must_on] == 0)
        illegal_off = np.any(action[self.must_off] == 1)
        if any([illegal_on, illegal_off]):
            return False
        else:
            return True

    def identify_must_ON_and_must_OFF_units(self):
        self.must_on = np.array(
            [True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array(
            [True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])
                  
        self.must_on_chp = np.array(
            [True if 0 < self.CHP_status[i] < self.t_min_up_CHP[i] else False for i in range(self.num_chp)])
        self.must_off_chp = np.array(
            [True if -self.t_min_down_CHP[i] < self.CHP_status[i] < 0 else False for i in range(self.num_chp)])
                     
        self.must_on_heateb = np.array(
            [True if 0 < self.HeatEB_status[i] < self.t_min_up_HeatEB[i] else False for i in range(self.num_heateb)])
        self.must_off_heateb = np.array(
            [True if -self.t_min_down_HeatEB[i] < self.HeatEB_status[i] < 0 else False for i in range(self.num_heateb)])
                      

    def update_gen_status(self, action_coal, action_chp, action_eb):

                                                     
        def single_update(status, action):
            if status > 0:
                if action == 1:
                    return (status + 1)
                else:
                    return -1
            else:
                if action == 1:
                    return 1
                else:
                    return (status - 1)

        def single_update_CHP(CHP_status, action_chp):
            if CHP_status > 0:
                if action_chp == 1:
                    return (CHP_status + 1)
                else:
                    return -1
            else:
                if action_chp == 1:
                    return 1
                else:
                    return (CHP_status - 1)

        def single_update_HeatEB(HeatEB_status, action_heateb):
            if HeatEB_status > 0:
                if action_heateb == 1:
                    return (HeatEB_status + 1)
                else:
                    return -1
            else:
                if action_heateb == 1:
                    return 1
                else:
                    return (HeatEB_status - 1)

        self.status = np.array([single_update(self.status[i], action_coal[i]) for i in range(len(self.status))])
        self.CHP_status = np.array(
            [single_update_CHP(self.CHP_status[i], action_chp[i]) for i in range(len(self.CHP_status))])
        self.HeatEB_status = np.array(
            [single_update_HeatEB(self.HeatEB_status[i], action_eb[i]) for i in range(len(self.HeatEB_status))])

                           
                        
    def prod_cost_env(self, disp, disp_c):
        cost_CO2 = 0
        cost_SO2 = 0
        cost_NOX = 0
        power_consumption_CO2 = 326.31
        Natural_gas_CO2 = 203.74
        power_consumption_SO2 = 3.14
        Natural_gas_SO2 = 0.011
        power_consumption_NOX = 1.134
        Natural_gas_NOX = 0.202
        EVS_CO2 = 0.023
        EVS_SO2 = 6
        EVS_NOX = 8
        ECS_CO2 = 0.01
        ECS_SO2 = 1
        ECS_NOX = 2
        for t1 in range(len(disp)):
            cost_CO2 = cost_CO2 + power_consumption_CO2 * disp[t1] * (EVS_CO2 + ECS_CO2)
            cost_SO2 = cost_SO2 + power_consumption_SO2 * disp[t1] * (EVS_SO2 + ECS_SO2)
            cost_NOX = cost_NOX + power_consumption_NOX * disp[t1] * (EVS_NOX + ECS_NOX)

        for t2 in range(len(disp_c)):
            cost_CO2 = cost_CO2 + Natural_gas_CO2 * disp_c[t2] * (EVS_CO2 + ECS_CO2)
            cost_SO2 = cost_SO2 + Natural_gas_SO2 * disp_c[t2] * (EVS_SO2 + ECS_SO2)
            cost_NOX = cost_NOX + Natural_gas_NOX * disp_c[t2] * (EVS_NOX + ECS_NOX)

        cost_env = cost_CO2 + cost_SO2 + cost_NOX
        return cost_env

    def calculate_gas_cost(self, gas, gas_sources):
        total_cost = []
        remaining_gas = gas.copy()                                          
        for i in range(len(gas)):
            total_gas_needed = remaining_gas[i]
            cost_for_this_p_value = 0
                                                                                  
            sorted_sources = sorted(gas_sources, key=lambda x: x['cost'])
            for source in sorted_sources:
                if total_gas_needed <= 0:
                    break
                gas_from_source = min(source['maxgas'], total_gas_needed)
                cost_for_this_p_value += gas_from_source * source['cost']
                total_gas_needed -= gas_from_source
            total_cost.append(cost_for_this_p_value)
        return np.array(total_cost)

    def prod_cost_funs(self, loads_vec: np.ndarray):
                              
        prod_cost_funs_vec = self.a * loads_vec ** 2 + self.b * loads_vec + self.c
                                 
        return np.where(loads_vec > 0, 1, 0) * prod_cost_funs_vec

    def prod_cost_funs_EB(self, loads_vec: np.ndarray):
                              
        prod_cost_funs_vec = self.operationalcost * loads_vec
                                 
        return np.where(loads_vec > 0, 1, 0) * prod_cost_funs_vec

    def prod_cost_funs_CHP(self, loads_vec: np.ndarray):
                                            
        Q_LHV = 9.7                       
        eta = 0.7
        gas = (1 / Q_LHV) * (loads_vec / eta)
        gas_sources = [
            {'gasbus': 1, 'maxgas': 24, 'cost': 25},
            {'gasbus': 2, 'maxgas': 72, 'cost': 25},
            {'gasbus': 3, 'maxgas': 23, 'cost': 25},
            {'gasbus': 4, 'maxgas': 14, 'cost': 25}
        ]

        cost = self.calculate_gas_cost(gas, gas_sources)
                                 
        return np.where(loads_vec > 0, 1, 0) * cost

                          
                                                                
    def _generator_fuel_costs(self, output, commitment):
        costs = 0
        for j in range(54):
            costs += (self.a[j] * output[j] ** 2 + self.b[j] * output[j] + self.c[j])
                                                                                               
                                                  
                                                                                                                 
                                                                            
                                    
                                                                
                                                           
                      
        return costs

    def calculate_lost_load_cost(self, net_demand, disp, disp_c, disp_e):
        diff = max(net_demand - np.sum(disp) - np.sum(disp_c) + np.sum(disp_e),
                   0)                                                            
                                          
                           
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount * 100
        return ens_cost, ens_amount

                                           
    def calculate_fuel_cost_and_dispatch(self, demand, commitment_coal, commitment_chp, commitment_eb):
                                                              
                           
                          
        disp, disp_c, disp_e, Penalty, cons, on_cost = self.economic_dispatch(commitment_coal, commitment_chp,
                                                                              commitment_eb, demand)
                                    
        env_costs = self.prod_cost_env(disp, disp_c)*0.1
        fuel_costs = self._generator_fuel_costs(disp, commitment_coal)
        fuel_costs_CHP = self.prod_cost_funs_CHP(disp_c)
        fuel_costs_eb = self.prod_cost_funs_EB(disp_e)
        fuel_costs_all = fuel_costs + np.sum(fuel_costs_CHP) + np.sum(fuel_costs_eb)
                                            
        return fuel_costs_all, env_costs, disp, disp_c, disp_e, Penalty, cons, on_cost

                                          
    def _get_net_demand(self):
        self.episode_timestep += 1
        demand_real = self.episode_forecast[self.episode_timestep]
        demand_heat = self.episode_heat[self.episode_timestep]
        demand_gas = self.episode_gas[self.episode_timestep]
        if self.episode_timestep < 287:
            self.load = self.day_load[self.episode_timestep + 1, :]
        else:
            self.load = self.day_load[self.episode_timestep, :]
                                           
        self.demand_real = demand_real
                                                               
        net_demand = demand_real
        net_demand = np.clip(net_demand, self.min_demand, self.max_demand)
        return net_demand, demand_heat, demand_gas

                         
    def _get_reward(self):
                                                                                         
        operating_cost = self.fuel_cost * average_demand / self.net_demand + self.Penalty + self.ens_cost + self.on_cost + self.env_cost
        reward = -operating_cost
        self.reward = reward
                       
        return reward

              
    def summarize_marginal_functions(self):
                                  
                          
                                     
                               
                        
                                      
                                                                   
                                                                   
                                    
                      
                                     
                                              
                                           
                                          
                                         
                                                                                                      
                                                          
                                     
                     
        max_prod_cost_points_vec = np.where(
            self.prod_cost_funs(self.p_min_vec) > self.prod_cost_funs(self.p_max_vec),
            self.p_min_vec, self.p_max_vec)
        max_prod_cost_points_CHP_vec = np.where(
            self.prod_cost_funs_CHP(self.Min_output_CHP_vec) > self.prod_cost_funs_CHP(self.Max_output_CHP_vec),
            self.Min_output_CHP_vec, self.Max_output_CHP_vec)
        max_prod_cost_points_eb_vec = np.where(
            self.prod_cost_funs_EB(self.min_output_Heateb_vec) > self.prod_cost_funs_EB(self.max_output_Heateb_vec),
            self.min_output_Heateb_vec, self.max_output_Heateb_vec)

                       
        self.max_prod_costs_vec = self.prod_cost_funs(max_prod_cost_points_vec)
        self.max_prod_costs_CHP_vec = self.prod_cost_funs_CHP(max_prod_cost_points_CHP_vec)
        self.max_prod_costs_eb_vec = self.prod_cost_funs_EB(max_prod_cost_points_eb_vec)

                          
        self.min_prod_costs_MW_vec = self.max_prod_costs_vec / self.p_max_vec

        self.min_prod_costs_MW_CHP_vec = self.max_prod_costs_CHP_vec / self.Max_output_CHP_vec
        self.min_prod_costs_MW_eb_vec = self.max_prod_costs_eb_vec / self.max_output_Heateb_vec
                    

    def determine_priority_orders(self):
                                   
        up_times_vec = np.maximum(self.t_min_up, 0.001)
        up_times_CHP_vec = np.maximum(self.t_min_up_CHP, 0.001)
        up_times_eb_vec = np.maximum(self.t_min_up_HeatEB, 0.001)
                         
        ON_costs_vec = self.min_prod_costs_MW_vec / up_times_vec
        ON_costs_CHP_vec = self.min_prod_costs_MW_CHP_vec / up_times_CHP_vec
        ON_costs_eb_vec = self.min_prod_costs_MW_eb_vec / up_times_eb_vec
        self.ON_priorities_vec = ON_costs_vec
        self.ON_priorities_CHP_vec = ON_costs_CHP_vec
        self.ON_priorities_eb_vec = ON_costs_eb_vec
        self.ON_priorities_vec = np.concatenate((self.ON_priorities_vec, self.ON_priorities_CHP_vec))
                   
        self.ON_priority_idx_vec = self.ON_priorities_vec.argsort()
        self.ON_priority_eb_idx_vec = self.ON_priorities_eb_vec.argsort()
                    

                             
    def ensure_action_legitimacy(self, demand: float, action_coal: np.ndarray, action_chp: np.ndarray,
                                 action_eb: np.ndarray):
                                                            
        action_coal_vec = self._legalise_action_coal(action_coal)
        action_chp_vec = self._legalise_action_chp(action_chp)
        action_eb_vec = self._legalise_action_eb(action_eb)
                                                                    
                                       
                                                 
                                                            
                                
                                                                 
                                         
                            
                               
        t = np.sum(
            (1 + self.HeatEfficiency / self.ElectricalEfficiency) * self.Max_output_CHP_vec * action_chp_vec) + np.sum(
            action_coal_vec * self.p_max_vec)
        if t - self.demand_heat < demand:
            action_coal_vec, action_chp_vec = self._adjust_low_capacity(demand, action_coal_vec, action_chp_vec)
                                                        
                       
                                            
        elif np.sum(action_coal_vec * self.p_min_vec) + np.sum((
                                                                       1 + self.HeatEfficiency / self.ElectricalEfficiency) * self.Min_output_CHP_vec * action_chp_vec) - self.demand_heat > demand:
            action_coal_vec, action_chp_vec = self._adjust_excess_capacity(demand, action_coal_vec, action_chp_vec)
                                                                       
        if np.sum(action_chp_vec * self.Max_output_CHP_vec) + np.sum(
                action_eb * self.max_output_Heateb_vec) < self.demand_heat:

            action_chp_vec, action_eb_vec = self._adjust_low_capacity_heat(self.demand_heat, action_chp_vec,
                                                                           action_eb_vec)
                                                                       
                                            
        elif np.sum(action_chp_vec * self.Min_output_CHP_vec) + np.sum(
                action_eb_vec * self.min_output_Heateb_vec) > self.demand_heat:
            action_chp_vec, action_eb_vec = self._adjust_excess_capacity_heat(self.demand_heat, action_chp_vec,
                                                                              action_eb_vec)
                                                                       

                            
        return action_coal_vec, action_chp_vec, action_eb_vec

    def _check_for_future_demands(self, action_vec: np.ndarray):
                                  
        commits_vec = np.logical_and(np.logical_not(self.must_off),
                                     np.logical_and(np.logical_not(self.must_on),
                                                    self.commits_vec == 1)) * 1
                    
        if np.any(commits_vec):
                         
            prev_ON_idx_vec = np.where(commits_vec == 1)[0]
                                
            priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec if i in prev_ON_idx_vec])
                     
            demands_vec = self.episode_forecast
                          
            for idx in priority_idx_vec:
                                    
                max_timestep = min(self.episode_timestep + self.t_min_down[idx], self.episode_length - 1)
                                       
                max_cap = np.sum(action_vec * self.p_max_vec)
                                            
                if np.any(max_cap < demands_vec[self.episode_timestep: max_timestep]):
                               
                    action_vec[idx] = 1
                           
        return action_vec

    def _adjust_low_capacity(self, demand: float, action_coal: np.ndarray, action_chp: np.ndarray):
                                 
        low_action_coal = action_coal.copy()
                     
        already_OFF_idx_vec = np.where(action_coal == 0)[0]
                      
        must_not_OFF_idx_vec = np.where(self.must_off == False)[0]
                     
        can_ON_idx_vec = np.intersect1d(already_OFF_idx_vec, must_not_OFF_idx_vec)
                     
        already_OFF_chp_idx_vec = np.where(action_chp == 0)[0]
                      
        must_not_OFF_chp_idx_vec = np.where(self.must_off_chp == False)[0]
                     
        can_ON_chp_idx_vec = np.intersect1d(already_OFF_chp_idx_vec, must_not_OFF_chp_idx_vec)
        can_ON_chp_idx_vec = can_ON_chp_idx_vec + 54
        can_ON_all_idx_vec = np.concatenate((can_ON_idx_vec, can_ON_chp_idx_vec))

                    
        if len(can_ON_all_idx_vec) > 0:
                                
            priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec if i in can_ON_all_idx_vec])
                     
            t2 = np.sum(
                (1 + self.HeatEfficiency / self.ElectricalEfficiency) * self.Max_output_CHP_vec * action_chp) + np.sum(
                action_coal * self.p_max_vec)
            remaining_supply = demand + self.demand_heat - np.sum(action_coal * self.p_max_vec) - np.sum(
                (1 + self.HeatEfficiency / self.ElectricalEfficiency) * self.Max_output_CHP_vec * action_chp)
                          
            for idx in priority_idx_vec:
                           
                if idx > 53:
                    action_chp[idx - 54] = 1
                               
                    remaining_supply = remaining_supply - self.Max_output_CHP_vec[idx - 54]
                else:
                    action_coal[idx] = 1
                               
                    remaining_supply = remaining_supply - self.p_max_vec[idx]
                           
                                        
                if remaining_supply <= 0.0001:
                    break
                           
        return action_coal, action_chp

    def _adjust_excess_capacity(self, demand: float, action_coal: np.ndarray, action_chp: np.ndarray):
                                 
        excess_action_vec = action_coal.copy()
                           
                     
        already_ON_idx_vec = np.where(action_coal == 1)[0]
                      
        must_not_ON_idx_vec = np.where(self.must_on == False)[0]
                     
        can_OFF_idx_vec = np.intersect1d(already_ON_idx_vec, must_not_ON_idx_vec)
        already_ON_chp_idx_vec = np.where(action_chp == 0)[0]
        must_not_ON_chp_idx_vec = np.where(self.must_off_chp == False)[0]
        can_OFF_chp_idx_vec = np.intersect1d(already_ON_chp_idx_vec, must_not_ON_chp_idx_vec)
        can_OFF_chp_idx_vec = can_OFF_chp_idx_vec + 54
        can_OFF_all_idx_vec = np.concatenate((can_OFF_idx_vec, can_OFF_chp_idx_vec))
                    
        if len(can_OFF_all_idx_vec) > 0:
                                    
            OFF_priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec[::-1] if i in can_OFF_all_idx_vec])
                      
                                                                          
            excess_supply = np.sum(action_coal * self.p_min_vec) + np.sum((
                                                                                  1 + self.HeatEfficiency / self.ElectricalEfficiency) * self.Min_output_CHP_vec * action_chp) - demand - self.demand_heat
                          
            for idx in OFF_priority_idx_vec:
                           
                if idx > 53:
                    action_chp[idx - 54] = 0
                                
                    excess_supply -= self.Min_output_CHP_vec[idx - 54]
                else:
                    action_coal[idx] = 0
                                
                    excess_supply -= self.p_min_vec[idx]
                                   
                if excess_supply <= 0.0001:
                                           
                    if np.sum(action_coal * self.p_max_vec) + np.sum((
                                                                             1 + self.HeatEfficiency / self.ElectricalEfficiency) * self.Max_output_CHP_vec * action_chp) - self.demand_heat < demand:
                        if idx > 53:
                            action_chp[idx - 54] = 1
                        else:
                            action_coal[idx] = 1
                        break
                    break
                           
        return action_coal, action_chp

    def _adjust_low_capacity_heat(self, demand_heat: float, action_chp: np.ndarray, action_eb: np.ndarray):
                                 
        low_action_coal = action_chp.copy()
                     
        already_OFF_idx_vec = np.where(action_eb == 0)[0]
                      
        must_not_OFF_idx_vec = np.where(self.must_off_heateb == False)[0]
                     
        can_ON_idx_vec = np.intersect1d(already_OFF_idx_vec, must_not_OFF_idx_vec)
                     
        already_OFF_eb_idx_vec = np.where(action_eb == 0)[0]
                      
        must_not_OFF_eb_idx_vec = np.where(self.must_off_heateb == False)[0]
                     
        can_ON_eb_idx_vec = np.intersect1d(already_OFF_eb_idx_vec, must_not_OFF_eb_idx_vec)

                    
        if len(can_ON_eb_idx_vec) > 0:
                                
            priority_idx_vec = np.array([i for i in self.ON_priority_eb_idx_vec if i in can_ON_eb_idx_vec])
                     
            remaining_supply = self.demand_heat - np.sum(
                action_chp * self.Max_output_CHP_vec) + np.sum(action_eb * self.max_output_Heateb_vec)
                          
            for idx in priority_idx_vec:
                           
                action_eb[idx] = 1
                           
                remaining_supply = remaining_supply - self.max_output_Heateb_vec[idx]
                           
                                        
                if remaining_supply <= 0.0001:
                    break
                           
        return action_chp, action_eb

    def _adjust_excess_capacity_heat(self, demand_heat: float, action_chp: np.ndarray, action_eb: np.ndarray):
                                 
        excess_action_vec = action_chp.copy()
                           
                     
        already_ON_idx_vec = np.where(action_eb == 1)[0]
                      
        must_not_ON_idx_vec = np.where(self.must_on_heateb == False)[0]
                     
        can_OFF_idx_vec = np.intersect1d(already_ON_idx_vec, must_not_ON_idx_vec)
                    
        if len(can_OFF_idx_vec) > 0:
                                    
            OFF_priority_idx_vec = np.array([i for i in self.ON_priority_eb_idx_vec[::-1] if i in can_OFF_idx_vec])
                      
            excess_supply = np.sum(action_chp * self.Min_output_CHP_vec) + np.sum(
                action_eb * self.min_output_Heateb_vec) - demand_heat
                          
            for idx in OFF_priority_idx_vec:
                           
                action_eb[idx] = 0
                            
                excess_supply -= self.min_output_Heateb_vec[idx]
                                   
                if excess_supply <= 0.0001:
                                           
                    if np.sum(action_chp * self.Max_output_CHP_vec) + np.sum(
                            action_eb * self.max_output_Heateb_vec) < demand_heat:
                        action_eb[idx] = 1
                        break
                    break
                           
        return action_chp, action_eb

                              
    def economic_dispatch(self, commitment_coal, commitment_chp, commitment_eb, demand):
                   
                                       
                              
                               
                
        operationalcost_eb = self.operationalcost
        demand_heat = self.demand_heat
        idx = np.where(np.array(commitment_coal) == 1)[0]
        on_a = self.a[idx]
        on_b = self.b[idx]
        on_min = self.min_output[idx]
        on_max = self.max_output[idx]
                    
                           
        power = self.power[idx]
        disp = np.zeros(self.num_gen)                        
        rampup = self.RampUp[idx]
        rampdown = self.RampDown[idx]
        idx_c = np.where(np.array(commitment_chp) == 1)[0]
        on_min_c = self.Min_output_CHP[idx_c]
        on_max_c = self.Max_output_CHP[idx_c]
        power_c = self.power_c[idx_c]
        disp_c = np.zeros(self.num_chp)
        rampup_c = self.RampUp_CHP[idx_c]
        rampdown_c = self.RampDown_CHP[idx_c]
        disp_e = np.zeros(self.num_heateb)
        idx_e = np.where(np.array(commitment_eb) == 1)[0]
        on_min_e = self.min_output_Heateb[idx_e]
        on_max_e = self.max_output_Heateb[idx_e]
        power_e = self.power_e[idx_e]
        Penalty = 0
        start = time.time()
        econ_coal, econ_chp, econ_eb, result_gas = ipopt_solve(demand, demand_heat, on_a, on_b, on_min, on_max, rampup,
                                                               rampdown, power, on_min_c, on_max_c, rampup_c,
                                                               rampdown_c,
                                                               power_c, on_min_e, on_max_e, power_e, operationalcost_eb)
        end = time.time()
                                               

        cons = self.check_constraints(econ_coal, power, rampup, rampdown, on_min, on_max)
                                                                   
                            
        n_g = len(on_a)
        for g in range(n_g):
                        
                              
            if econ_coal[g] - power[g] > rampup[g] + 0.1:
                Penalty += 100
            if econ_coal[g] - power[g] < -rampdown[g] - 0.1:
                Penalty += 100

        for g in range(n_g):
            if econ_coal[g] < on_min[g] - 1:
                Penalty += 500
            if econ_coal[g] > on_max[g] + 1:
                Penalty += 500
        disp[idx] = econ_coal
        disp_c[idx_c] = econ_chp
        disp_e[idx_e] = econ_eb
        self.power = disp
        self.power_c = disp_c
        self.power_e = disp_e
        on_cost = n_g * 0

        return disp, disp_c, disp_e, Penalty, cons, on_cost

                             
    def onoff(self, unit):
        num_gen = len(unit)
        for i in range(num_gen):
            if unit[i] < 1:
                unit[i] = 0
            else:
                unit[i] = 1
        return unit

    def check_constraints(self, econ, power, rampup, rampdown, min_on, max_on):
        num_gen = len(econ)
        cons = 0
        cons_demand = 0
        for g in range(num_gen):
            if power[g] > 0.1 and econ[g] > 0.1:
                            
                if econ[g] - power[g] > rampup[g] + 0.1:
                    cons += 1
                                       
                            
                if econ[g] - power[g] < -rampdown[g] - 0.1:
                    cons += 1
                                             
                if econ[g] > max_on[g]:
                    cons_demand += 1
                            
                if econ[g] < min_on[g]:
                    cons_demand += 1
        return cons

    def _transition(self, action):
        action_vec = action
        self.net_demand, self.demand_heat, self.demand_gas = self._get_net_demand()
                
        state = self.get_current_state()
                       
        action_coal = action_vec[:54]
                          
        action_chp = action_vec[54:63]
                      
        action_eb = action_vec[63:]
                  
                                       
                                      
                                    
                                    
                                     
                                     
                                              
        action_coal_vec, action_chp_vec, action_eb_vec = self.ensure_action_legitimacy(self.net_demand, action_coal,
                                                                                       action_chp, action_eb)
                                             
                                                     
        action_coal_vec = np.array(action_coal_vec)
        action_chp_vec = np.array(action_chp_vec)
        action_eb_vec = np.array(action_eb_vec)
                                        
                                                         
        self.fuel_cost, self.env_cost, self.disp, self.disp_c, self.disp_e, self.Penalty, self.cons, self.on_cost = self.calculate_fuel_cost_and_dispatch(
            self.net_demand, action_coal_vec, action_chp_vec, action_eb_vec)
        self.ens_cost, self.ens_amount = self.calculate_lost_load_cost(self.net_demand, self.disp, self.disp_c,
                                                                       self.disp_e)
                                          
                                                                                                            
                                                  
                               
                                               
        self.day_cost += self.fuel_cost
                      
        state = self.get_next_state(action_coal_vec, action_chp_vec, action_eb_vec)
        return state

    def step(self, action):
        obs = self._transition(action)                                                   
                                                       
        reward = self._get_reward()                                     
        done = self.is_terminal()                                    

        return obs, reward, done

                                     
    def is_terminal(self):
        if self.mode == "train":
                                                                                     
            a = self.episode_timestep == (self.episode_length - 1)
            return a
        else:
            return self.episode_timestep == (self.episode_length - 1)

                                                     
    def sample_day(self):
        day = np.random.choice(self.profiles_df.date, 1)
        day_profile = self.profiles_df[self.profiles_df.date == day.item()]
        formatted_dates = pd.to_datetime(day).strftime('%Y-%m-%d')
        day1 = np.array(formatted_dates)
        day0 = day[0]
        date_obj = datetime.strptime(day0, '%m/%d/%Y')
        day2 = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
        day2 = np.array(day2)
        day_load = self.busload_df[self.busload_df.date == day2.item()]
                                                                                           
                                    
                                                                                             
                                                                       
        init_power = np.array(
            [-0.0019629928, -0.0023276387, -0.004338639, 0.0016420172, 324.8588, 61.354877, 0.003591261, 0.005071797,
             -0.0073459097, -0.004845536, 158.79825, 226.66078, 0.00068357866, -0.0037892177, -0.00485382, -0.002897333,
             -0.0020420188, 0.0059631895, -0.007838323, 19.99522, 147.25014, 34.65255, -0.0056839585, 0.0029621462,
             111.8776, 115.48593, -0.00691861, 282.25772, 282.97867, 372.82516, -0.007936818, -0.0016285338,
             0.00095338817, 0.00089761615, 0.0061391313, 0.003128313, 344.36905, 0.0018798842, -0.0025446303, 438.26175,
             0.0016877323, 0.0027994784, 0.00016357866, -0.00085145375, 181.89182, 28.871616, -0.0044776006,
             -0.0005087219, -0.0037688985, 0.0041993065, 25.986645, -0.0051801433, 0.0055266013, 0.00247274])
        return day, day_profile, init_power, day_load

    def _update_production_capacities(self, action_vec, action_chp, power, power_c):
                       
        p_min_vec = self.min_output
        p_max_vec = self.max_output
        min_output_CHP_vec = self.Min_output_CHP
        max_output_CHP_vec = self.Max_output_CHP
                                    
                    
                    
        self.p_min_vec = np.maximum(p_min_vec, action_vec * (power - self.RampDown))
                    
        self.p_max_vec = np.minimum(p_max_vec, action_vec * (power + self.RampUp)
                                    + np.where(action_vec == 0, 1, 0) * p_max_vec)
        self.p_max_vec = np.maximum(self.p_max_vec, self.p_min_vec + 0.1)
        self.Min_output_CHP_vec = np.maximum(min_output_CHP_vec, action_chp * (power_c - self.RampDown_CHP))
        self.Max_output_CHP_vec = np.minimum(max_output_CHP_vec, action_chp * (power_c + self.RampUp_CHP)
                                             + np.where(action_chp == 0, 1, 0) * max_output_CHP_vec)
        self.Max_output_CHP_vec = np.maximum(self.Max_output_CHP_vec, self.Min_output_CHP_vec + 0.1)
                                     
        if np.any(self.p_min_vec > self.p_max_vec):
            raise Exception("Min capacity > Max capacity.")
        if np.any(self.Min_output_CHP_vec > self.Max_output_CHP_vec):
            raise Exception("Min capacity chp > Max capacity chp.")

    def _update_production_capacities_init(self, action_vec, action_chp, power, power_c):
                       
        p_min_vec = self.min_output
        p_max_vec = self.max_output
        min_output_CHP_vec = self.Min_output_CHP
        max_output_CHP_vec = self.Max_output_CHP
                    
        self.p_min_vec = np.maximum(p_min_vec, action_vec * (power - self.RampDown))
                    
        self.p_max_vec = np.minimum(p_max_vec, action_vec * (power + self.RampUp)
                                    + np.where(action_vec == 0, 1, 0) * p_max_vec)
        self.p_max_vec = np.maximum(self.p_max_vec, self.p_min_vec + 0.1)
        self.Min_output_CHP_vec = np.maximum(min_output_CHP_vec, action_chp * (power_c - self.RampDown_CHP))
        self.Max_output_CHP_vec = np.minimum(max_output_CHP_vec, action_chp * (power_c + self.RampUp_CHP)
                                             + np.where(action_chp == 0, 1, 0) * max_output_CHP_vec)
        self.Max_output_CHP_vec = np.maximum(self.Max_output_CHP_vec, self.Min_output_CHP_vec + 0.1)
                                     
        if np.any(self.p_min_vec > self.p_max_vec):
            raise Exception("Min capacity > Max capacity.")
        if np.any(self.Min_output_CHP_vec > self.Max_output_CHP_vec):
            raise Exception("Min capacity chp > Max capacity chp.")

    def reset(self):
        day, day_profile, init_power, day_load = self.sample_day()
        self.day = day
        self.episode_forecast = day_profile.demand.values

        self.episode_heat = day_profile.heatLoad.values
        self.episode_heat = self.episode_heat * 2
        self.episode_gas = day_profile.gasload.values
        self.episode_solar_forecast = day_profile.solar.values
        self.day_load = day_load.iloc[:, 2:].values
                         
                              
        self.episode_forecast = self.episode_forecast-self.episode_solar_forecast
                                     
        self.episode_timestep = -1
        self.net_demand = None
        self.day_cost = 0
        self.fuel_cost = 0
                                               
        self.status = self.gen_info['status'].to_numpy()
        self.CHP_status = self.CHP_info['status'].to_numpy()            
        self.HeatEB_status = self.HeatEB_info['status'].to_numpy()
                                 
        self.expected_cost = 0
        self.ens = False
        self.ens_amount = 0
        self.cons = 0
                                      
        self.load = self.day_load[self.episode_timestep + 1, :]
        self.status, action_init = check_status(init_power, self.status)
        action_init_vec = np.array(action_init)
        self.commits_vec = np.where(self.status > 0, 1, 0)
        self._determine_constraints()
        self.power = check_power(init_power)
        self.power_c = np.ones(self.num_chp) * 30
        self.CHP_status, action_init_chp = check_status(self.power_c, self.CHP_status)
        self.HeatEB_status_status, action_init_eb = check_status(self.power_c, self.CHP_status)
        self.power_e = np.zeros(self.num_heateb)
        self._update_production_capacities_init(action_init_vec, action_init_chp, self.power,
                                                self.power_c)
        state = self._get_state()
        return state


def check_status(power, status):
    num = len(power)
    action = [0] * num
                        
    for i in range(num):
        if power[i] > 5:
            status[i] = 1
            action[i] = 1
        elif power[i] < 1:
            status[i] = -10
        else:
            status[i] = 0
    return status, action


def check_power(power):
    num = len(power)
    for i in range(num):
        if power[i] < 0:
            power[i] = 0
    return power


def create_gen_info(num_gen, dispatch_freq_mins):
    MIN_GENS = 5
    if num_gen < 5:
        raise ValueError("num_gen should be at least {}".format(MIN_GENS))
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gen6 = pd.read_csv(os.path.join(script_dir, Gen_units_path))
    gen_info = gen6                                                   
    gen_info = gen_info.sort_index()
    gen_info.reset_index()
                                  
    gen_info.t_min_up = gen_info.t_min_up
    gen_info.t_min_down = gen_info.t_min_down
    gen_info.status = gen_info.status
    gen_info = gen_info.astype({'t_min_down': 'int64',
                                't_min_up': 'int64',
                                'status': 'int64'})
                                                                     
                          
    return gen_info


class UnitCommitmentEnv(gym.Env):
    def __init__(self, **kwargs):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        env_fn = os.path.join(script_dir, 'data/envs/6gen.json')
        params = json.load(open(env_fn))
        gen_info = create_gen_info(params.get('num_gen', DEFAULT_NUM_GEN),
                                   params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
        CHP_info = pd.read_csv(os.path.join(script_dir, CHP_units_path))
        HeatEB_info = pd.read_csv(os.path.join(script_dir, HeatEB_units_path))
        profiles_df = pd.read_csv(os.path.join(script_dir, ParameterData_train_demand))
        busload_df = pd.read_csv(os.path.join(script_dir, busload))
                            
        self.env = Env(gen_info=gen_info, CHP_info=CHP_info, HeatEB_info=HeatEB_info, profiles_df=profiles_df,
                       busload_df=busload_df, mode='train',
                       **kwargs)

                     
        self.observation_space = spaces.Dict({
            'status': spaces.Box(low=-(np.ones(self.env.num_gen) * 10000),
                                 high=np.ones(self.env.num_gen) * 10000,
                                 dtype=np.int64),
            'CHP_status': spaces.Box(low=-(np.ones(self.env.num_gen) * 10000),
                                     high=np.ones(self.env.num_gen) * 10000,
                                     dtype=np.int64),
            'HeatEB_status': spaces.Box(low=-(np.ones(self.env.num_gen) * 10000),
                                        high=np.ones(self.env.num_gen) * 10000,
                                        dtype=np.int64),
            'load': spaces.Box(low=-(np.ones(30) * 1000),
                               high=np.ones(30) * 1000,
                               dtype=np.int64),
            'demand_forecast': spaces.Box(low=0, high=np.inf, shape=(self.env.episode_length,), dtype=np.float32),
            'demand_heat': spaces.Box(low=0, high=np.inf, shape=(self.env.episode_length,), dtype=np.float32),
            'demand_gas': spaces.Box(low=0, high=np.inf, shape=(self.env.episode_length,), dtype=np.float32),
            'solar_forecast': spaces.Box(low=0, high=np.inf, shape=(self.env.episode_length,), dtype=np.float32),
            'cost': spaces.Box(low=0, high=np.inf, shape=(self.env.num_gen,), dtype=np.float32),
            'timestep': spaces.Discrete(self.env.episode_length),
            'power': spaces.Box(low=np.zeros(self.env.num_gen), high=np.ones(self.env.num_gen) * 1000,
                                dtype=np.float32),
            'power_c': spaces.Box(low=np.zeros(self.env.num_chp), high=np.ones(self.env.num_chp) * 1000,
                                  dtype=np.float32),
            'power_e': spaces.Box(low=np.zeros(self.env.num_heateb), high=np.ones(self.env.num_heateb) * 1000,
                                  dtype=np.float32),
            'day_cost': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'day': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),
            'ens': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'cons': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64)
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(self.env.num_gen,), dtype=np.int8)

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        pass

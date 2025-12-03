import os
import random
import sys
import time

import wntr
from wntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
from s2v_wdn_dqn.envs.wdn_env import WDNEnv

def create_water_network_model():
    # 1. Create a new water network model
    wn = wntr.network.WaterNetworkModel()

    wn.options.hydraulic.demand_model = 'PDD'

    pattern_house1 = [1.0]*8 + [5.0]*8 + [1.0]*8  
    pattern_house2 = [1.0]*8 + [1.0]*8 + [5.0]*8  
    pattern_house3 = [5.0]*8 + [1.0]*8 + [1.0]*8  
    slow_pattern_house = [1.0]*8 + [2.0]*12 + [1.0]*4    
    pump_speed_pattern = [1.0]*24

    #tank_head_pattern = [1.0]*6 + [0.5]*6 + [1.0]*6 + [0.5]*6
    
    
    wn.add_pattern('slow_pattern_house', slow_pattern_house)
    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)
    wn.add_pattern('pump_speed_pattern', pump_speed_pattern)
    #wn.add_pattern('tank_head_pattern', tank_head_pattern)


    wn.add_curve('Pump1_Curve', 'HEAD' , [(0, 50), (5, 45), (10, 40)])  # Head vs. Flow
    wn.add_curve('Pump2_Curve', 'HEAD' , [(0, 90), (5, 85), (10, 80)])  # Head vs. Flow



    # 2. Add a Reservoir (Tank1) on the left
    wn.add_reservoir('R1', base_head=200.0, head_pattern=None, coordinates=(-50, 50))
    #wn.add_tank('R1', elevation=50, init_level=100, max_level=5000, min_level=0.0, coordinates=(-50, 50))

    # 3. Build a rectangular loop (9 junctions: J0–J8)
    wn.add_junction('J0', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 100))
    wn.add_junction('J1', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 100))
    wn.add_junction('J2', base_demand=0.0, elevation=100.0, demand_pattern=None, coordinates=(100, 100))
    wn.add_junction('J3', base_demand=0.0, elevation=100.0, demand_pattern=None, coordinates=(100, 50))
    wn.add_junction('J4', base_demand=0.0, elevation=100.0, demand_pattern=None, coordinates=(100, 0))
    wn.add_junction('J5', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 0))
    wn.add_junction('J6', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 0))
    wn.add_junction('J7', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 50))
    wn.add_junction('J8', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 50))

    # 4. Replace the reservoir connection with a pump:
    wn.add_pipe('P_R1_J7', 'R1', 'J7', length=50, diameter=0.6, roughness=100, minor_loss=0)
    #wn.add_pump('P1', 'R1', 'J7', initial_status='OPEN', pump_type='HEAD', pump_parameter='Pump1_Curve', speed=1.0)
    #wn.add_pump('P1', 'R1', 'J7', initial_status='OPEN', pump_type='POWER', pump_parameter=0.1)

    # 5. Connect the 8 junctions in a loop (rectangle)
    wn.add_pipe('PR0', 'J0', 'J1', length=50, diameter=0.6, roughness=100 , minor_loss=0)
    #wn.add_pipe('PR1', 'J1', 'J2', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pump('P1',  'J1', 'J2', initial_status='OPEN', pump_type='POWER', speed=1)
    wn.add_pipe('PR2', 'J2', 'J3', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR3', 'J3', 'J4', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR4', 'J5', 'J4', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR5', 'J6', 'J5', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR6', 'J7', 'J6', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR7', 'J7', 'J0', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR8', 'J7', 'J8', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR9', 'J8', 'J3', length=50, diameter=0.6, roughness=100, minor_loss=0)

    # 6. Add three houses (H1, H2, H3) branching from the right side
    wn.add_junction('H1', base_demand=0.5, elevation=10.0, demand_pattern=None, coordinates=(120, 100))
    wn.add_junction('H2', base_demand=0.5, elevation=10.0, demand_pattern=None, coordinates=(120, 50))
    wn.add_junction('H3', base_demand=0.5, elevation=10.0, demand_pattern=None, coordinates=(120, 0))
    #wn.add_junction('H1', base_demand=0.1, elevation=150.0, coordinates=(120, 100))
    #wn.add_junction('H2', base_demand=0.1, elevation=150.0, coordinates=(120, 50))
    #wn.add_junction('H3', base_demand=0.1, elevation=150.0, coordinates=(120, 0))

    # 7. Connect houses to the loop.
    wn.add_pipe('PH1', 'J2', 'H1', length=20, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PH3', 'J4', 'H3', length=20, diameter=0.6, roughness=100, minor_loss=0)

    # For H2, remove the existing pipe and add a valve instead:
    wn.add_pipe('PH2', 'J3', 'H2', length=20, diameter=0.6, roughness=100, minor_loss=0)
    #wn.add_valve('V1', 'J3', 'H2', diameter=0.3, valve_type="FCV", initial_status='Active')

    return wn


def create_new_water_model():

    wn = wntr.network.WaterNetworkModel()
    
    pattern_house1 = [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0]  
    pattern_house2 = [0.5]*7 + [1.5]*4 + [0.5]*6 + [1.5]*4 + [0.5]*3  
    pattern_house3 = [2.5]*8 + [0.0]*12 + [2.5]*4    

    slow_pattern_house = [1.0]*8 + [2.0]*12 + [1.0]*4    
    pump_speed_pattern = [1.0]*24

    wn.add_pattern('slow_pattern_house', slow_pattern_house)
    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)
    wn.add_pattern('pump_speed_pattern', pump_speed_pattern)

    wn.options.hydraulic.demand_model = 'PDD'  # Pressure-driven demand
    wn.add_reservoir('R1', base_head=100, coordinates=(0, 50))
    wn.add_tank('T1', elevation=30, init_level=5, min_level=2, max_level=10, diameter=10, coordinates=(100, 50))

    junction_data = {
        'J1': {'demand': 0.05, 'elevation': 10, 'coordinates': (50, 40)},
        'J2': {'demand': 0.04, 'elevation': 12, 'coordinates': (75, 30)},
        'J3': {'demand': 0.03, 'elevation': 15, 'coordinates': (100, 20)},
        'J4': {'demand': 0.06, 'elevation': 8, 'coordinates': (120, 40)},
    }
    for j, data in junction_data.items():
        wn.add_junction(j, base_demand=data['demand'], elevation=data['elevation'], coordinates=data['coordinates'])

    wn.add_pipe('P1', 'R1', 'J1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P3', 'J2', 'J3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P5', 'J4', 'T1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P6', 'J2', 'T1', length=70, diameter=0.3, roughness=100)

    wn.add_curve('Pump1_Curve', 'HEAD' , [(0, 50), (5, 45), (10, 40)])  # Head vs. Flow
    wn.add_curve('Pump2_Curve', 'HEAD' , [(0, 90), (5, 85), (10, 80)])  # Head vs. Flow
    wn.add_pump('Pump1', 'J1', 'J2', pump_type='HEAD', pump_parameter='Pump1_Curve', speed=1.0)

    wn.add_valve('V1', 'J3', 'J4', valve_type='PRV', diameter=0.2, initial_setting=30)

    return wn


def random_simulation_test():
    one_day_in_seconds = 86400 / 12
    global_timestep = 5

    #wn = wntr.network.WaterNetworkModel('NET_2.inp')
    #wn = wntr.network.WaterNetworkModel('NET_4.inp')
    #wn = wntr.network.WaterNetworkModel('L-TOWN_Real.inp')
    #wn = wntr.network.WaterNetworkModel("Modelli Belmonte Castel Sant'Angelo_2024-09-05_0850/Modello_Castel'Sant'Angelo/_Modelli idraulici/Pacchetto_CSA/CSA_Base.inp")
    #wn = wntr.network.WaterNetworkModel("Modelli Belmonte Castel Sant'Angelo_2024-09-05_0850/Modello Belmonte/Modelli_MIKE/SDF/BELMONTE_DHIBase.inp")
    #wn = create_water_network_model()
    #wn.add_pattern('house1_pattern', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([1,5,1], global_timestep, simulation_duration=one_day_in_seconds))
    #wn.add_pattern('ptn_1', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([1,3,5,3,1], global_timestep, simulation_duration=one_day_in_seconds))

    #wn.get_node('R1').max_level = 600
    i = 0
    while i < 1:
        #wn = wntr.network.WaterNetworkModel('NET_4.inp')

        #wn.reset_initial_values()
        #wn = create_water_network_model()
        
        wn = wntr.network.WaterNetworkModel('./nodes_and_inps/L-TOWN_Real.inp')


        #wn.add_pattern('house1_pattern', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([1,5,1], global_timestep, simulation_duration=one_day_in_seconds))
        wn.add_pattern('ptn_1', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([1,3,5,3,1], global_timestep, simulation_duration=one_day_in_seconds))
        #sys.exit()

        sim = InteractiveWNTRSimulator(wn)
        sim.init_simulation(duration=one_day_in_seconds, global_timestep=global_timestep)

        #sim.plot_network(link_labels=False, node_labels=True)
        #return

        #branched_sim_1 = None
        #branched_sim_2 = None

        start = time.time()

        #sim.plot_network()
        sims = [sim]

        has_active_leak = []
        has_active_demand = []
        closed_pipe = []


        node_list = wn.junction_name_list
        link_list = wn.link_name_list

        try:
            while not sim.is_terminated():
                #print(f"Current time: {current_time} {current_time / sim.hydraulic_timestep()}")
                current_time = sim.get_sim_time()
                r = random.random()
                if r < 0.01:
                    r2 = random.random()
                    if r2 < 0.3:
                        if len(has_active_leak) == 0 or random.random() < 0.5:    
                            node = random.choice(node_list)
                            sim.start_leak(node, 0.1)
                            has_active_leak.append(node)
                            #print(f"Leak started on {node}")
                        else:
                            node = random.choice(has_active_leak)
                            sim.stop_leak(node)
                            has_active_leak.remove(node)
                            #print(f"Leak stopped on {node}")
                    elif r2 < 0.6:
                        if len(has_active_demand) == 0 or random.random() < 0.5:    
                            node = random.choice(node_list)
                            sim.change_demand(node, 1, name='ptn_1')
                            has_active_demand.append(node)
                            #print(f"Demand added on {node}")
                        else:
                            node = random.choice(has_active_demand)
                            sim.change_demand(node)
                            has_active_demand.remove(node)
                            #print(f"Demand removed on {node}")
                    else:
                        if len(closed_pipe) == 0 or random.random() < 0.5:    
                            link = random.choice(link_list)
                            sim.close_pipe(link)
                            closed_pipe.append(link)
                            #print(f"Pipe closed {link}")
                        else:
                            link = random.choice(closed_pipe)
                            sim.open_pipe(link)
                            closed_pipe.remove(link)
                            #print(f"Pipe opened {link}")

                #for s in sims:
                #    s.step_sim()
                sim.step_sim()

            end = time.time()
            #print(f"Elapsed time: {end - start}")
            
            
            if sim.get_sim_time() >= one_day_in_seconds - global_timestep:
                sim.plot_network(link_labels=False, node_labels=True)
                sim.plot_results('node', 'pressure')
                sim.plot_results('link', 'flowrate')
                sim.plot_results('node', 'demand')
                sim.plot_results('node', 'satisfied_demand')
                sim.plot_network_over_time(node_key='satisfied_demand', link_key='flowrate', node_labels=True, link_labels=False)
                i += 1
                print(f"Simulation {i} completed in {end - start} seconds.")
            else:
                print("Simulation terminated before reaching the end time.")

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                print("Simulation interrupted by user.")
                sys.exit()
            else:
                print(e)
                print("An error occurred during the simulation.")
            #for s in sims:
            #    s.dump_results_to_csv()
            #sys.exit()


def custom_complex_water_model():
    # Create a new water network model
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.demand_model = 'PDD'  # Pressure-driven demand


    wn.add_reservoir('R1', base_head=100,                coordinates=(0, 0))
    wn.add_junction('J1', base_demand=0.0, elevation=10, coordinates=(0  , 25))
    wn.add_junction('J2', base_demand=0.0, elevation=10, coordinates=(25 , 25))
    wn.add_junction('J3', base_demand=0.0, elevation=10, coordinates=(25 , 0 ))
    wn.add_junction('J4', base_demand=0.0, elevation=10, coordinates=(25 ,-25))
    wn.add_junction('J5', base_demand=0.0, elevation=10, coordinates=(0  ,-25))
    wn.add_junction('J6', base_demand=0.0, elevation=10, coordinates=(-25,-25))
    wn.add_junction('J7', base_demand=0.0, elevation=10, coordinates=(-25,  0))
    wn.add_junction('J8', base_demand=0.0, elevation=10, coordinates=(-25, 25))


    wn.add_junction('JS1-1', base_demand=0.0, elevation=10, coordinates=(0   , 50))
    wn.add_junction('JS2-1', base_demand=0.0, elevation=10, coordinates=(50 , 50))
    wn.add_junction('JS3-1', base_demand=0.0, elevation=10, coordinates=(50 , 0 ))
    wn.add_junction('JS4-1', base_demand=0.0, elevation=10, coordinates=(50 ,-50))
    wn.add_junction('JS5-1', base_demand=0.0, elevation=10, coordinates=(0   ,-50))
    wn.add_junction('JS6-1', base_demand=0.0, elevation=10, coordinates=(-50,-50))
    wn.add_junction('JS7-1', base_demand=0.0, elevation=10, coordinates=(-50,  0))
    wn.add_junction('JS8-1', base_demand=0.0, elevation=10, coordinates=(-50, 50))



    
    wn.add_pipe('PR1', 'R1', 'J1',length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PR2', 'R1', 'J3',length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PR3', 'R1', 'J5',length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PR4', 'R1', 'J7',length=50, diameter=0.3, roughness=100)

    wn.add_pipe('P2', 'J1', 'J2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P3', 'J2', 'J3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P4', 'J3', 'J4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P5', 'J4', 'J5', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P6', 'J5', 'J6', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P7', 'J6', 'J7', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P8', 'J7', 'J8', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P9', 'J8', 'J1', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS1', 'J1', 'JS1-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2', 'J2', 'JS2-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3', 'J3', 'JS3-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4', 'J4', 'JS4-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5', 'J5', 'JS5-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6', 'J6', 'JS6-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7', 'J7', 'JS7-1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8', 'J8', 'JS8-1', length=50, diameter=0.3, roughness=100)

    #wn.add_pump('PR1', 'R1', 'J1', pump_type='POWER', pump_parameter=0, speed=1.0)
    #wn.add_pump('PR2', 'R1', 'J3', pump_type='POWER', pump_parameter=0, speed=1.0)
    #wn.add_pump('PR3', 'R1', 'J5', pump_type='POWER', pump_parameter=0, speed=1.0)
    #wn.add_pump('PR4', 'R1', 'J7', pump_type='POWER', pump_parameter=0, speed=1.0)
    #
    #wn.add_valve('P2', 'J1', 'J2', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P3', 'J2', 'J3', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P4', 'J3', 'J4', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P5', 'J4', 'J5', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P6', 'J5', 'J6', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P7', 'J6', 'J7', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P8', 'J7', 'J8', diameter=0.3, valve_type='TCV', initial_setting=100)
    #wn.add_valve('P9', 'J8', 'J1', diameter=0.3, valve_type='TCV', initial_setting=100)


    #wn.add_pump('PPS1', 'J1', 'JS1-4', pump_type='POWER', pump_parameter=10000, speed=50.0)
    #wn.add_pump('PS2', 'J2', 'JS2-1', pump_type='POWER', pump_parameter=0, speed=0.0)
    #wn.add_pump('PS3', 'J3', 'JS3-1', pump_type='POWER', pump_parameter=0, speed=0.0)
    #wn.add_pump('PS4', 'J4', 'JS4-1', pump_type='POWER', pump_parameter=0, speed=0.0)
    #wn.add_pump('PS5', 'J5', 'JS5-1', pump_type='POWER', pump_parameter=0, speed=0.0)
    #wn.add_pump('PS6', 'J6', 'JS6-1', pump_type='POWER', pump_parameter=0, speed=0.0)
    #wn.add_pump('PS7', 'J7', 'JS7-1', pump_type='POWER', pump_parameter=0, speed=0.0)
    #wn.add_pump('PS8', 'J8', 'JS8-1', pump_type='POWER', pump_parameter=0, speed=0.0)


    wn.add_junction('JS8-2', base_demand=0.0, elevation=10, coordinates=(-45,  55))
    wn.add_junction('JS8-3', base_demand=0.0, elevation=10, coordinates=(-50, 55))
    wn.add_junction('JS8-4', base_demand=0.0, elevation=10, coordinates=(-55, 55))
    wn.add_junction('JS8-5', base_demand=0.0, elevation=10, coordinates=(-50, 60))

    wn.add_pipe('PS8-1', 'JS8-1', 'JS8-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8-2', 'JS8-1', 'JS8-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8-3', 'JS8-1', 'JS8-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8-4', 'JS8-3', 'JS8-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8-5', 'JS8-2', 'JS8-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS8-6', 'JS8-5', 'JS8-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8-7', 'JS8-5', 'JS8-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS8-8', 'JS8-5', 'JS8-4', length=50, diameter=0.3, roughness=100)

    ###############################################################################################

     # Section 1 (going up, based on JS1-1)
    wn.add_junction('JS1-2', base_demand=0.0, elevation=10, coordinates=(-5, 55))
    wn.add_junction('JS1-3', base_demand=0.0, elevation=10, coordinates=(0, 55))
    wn.add_junction('JS1-4', base_demand=0.0, elevation=10, coordinates=(5, 55))
    wn.add_junction('JS1-5', base_demand=0.0, elevation=10, coordinates=(0, 60))

    wn.add_pipe('PS1-1', 'JS1-1', 'JS1-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS1-2', 'JS1-1', 'JS1-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS1-3', 'JS1-1', 'JS1-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS1-4', 'JS1-3', 'JS1-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS1-5', 'JS1-2', 'JS1-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS1-6', 'JS1-5', 'JS1-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS1-7', 'JS1-5', 'JS1-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS1-8', 'JS1-5', 'JS1-4', length=50, diameter=0.3, roughness=100)


    wn.add_pump('PPS1', 'J1', 'JS1-4', pump_type='POWER', pump_parameter=5000, speed=10.0)


    # Section 2 (going up, based on JS2-1)
    wn.add_junction('JS2-2', base_demand=0.0, elevation=10, coordinates=(45, 55))
    wn.add_junction('JS2-3', base_demand=0.0, elevation=10, coordinates=(50, 55))
    wn.add_junction('JS2-4', base_demand=0.0, elevation=10, coordinates=(55, 55))
    wn.add_junction('JS2-5', base_demand=0.0, elevation=10, coordinates=(50, 60))

    wn.add_pipe('PS2-1', 'JS2-1', 'JS2-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2-2', 'JS2-1', 'JS2-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2-3', 'JS2-1', 'JS2-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2-4', 'JS2-3', 'JS2-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2-5', 'JS2-2', 'JS2-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS2-6', 'JS2-5', 'JS2-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2-7', 'JS2-5', 'JS2-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS2-8', 'JS2-5', 'JS2-4', length=50, diameter=0.3, roughness=100)

    # Section 3 (going right, based on JS3-1)
    wn.add_junction('JS3-2', base_demand=0.0, elevation=10, coordinates=(55, 5))
    wn.add_junction('JS3-3', base_demand=0.0, elevation=10, coordinates=(55, 0))
    wn.add_junction('JS3-4', base_demand=0.0, elevation=10, coordinates=(55, -5))
    wn.add_junction('JS3-5', base_demand=0.0, elevation=10, coordinates=(60, 0))

    wn.add_pipe('PS3-1', 'JS3-1', 'JS3-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3-2', 'JS3-1', 'JS3-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3-3', 'JS3-1', 'JS3-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3-4', 'JS3-3', 'JS3-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3-5', 'JS3-2', 'JS3-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS3-6', 'JS3-5', 'JS3-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3-7', 'JS3-5', 'JS3-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS3-8', 'JS3-5', 'JS3-4', length=50, diameter=0.3, roughness=100)

    # Section 4 (going down, based on JS4-1)
    wn.add_junction('JS4-2', base_demand=0.0, elevation=10, coordinates=(45, -55))
    wn.add_junction('JS4-3', base_demand=0.0, elevation=10, coordinates=(50, -55))
    wn.add_junction('JS4-4', base_demand=0.0, elevation=10, coordinates=(55, -55))
    wn.add_junction('JS4-5', base_demand=0.0, elevation=10, coordinates=(50, -60))

    wn.add_pipe('PS4-1', 'JS4-1', 'JS4-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4-2', 'JS4-1', 'JS4-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4-3', 'JS4-1', 'JS4-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4-4', 'JS4-3', 'JS4-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4-5', 'JS4-2', 'JS4-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS4-6', 'JS4-5', 'JS4-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4-7', 'JS4-5', 'JS4-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS4-8', 'JS4-5', 'JS4-4', length=50, diameter=0.3, roughness=100)

    # Section 5 (going down, based on JS5-1)
    wn.add_junction('JS5-2', base_demand=0.0, elevation=10, coordinates=(-5, -55))
    wn.add_junction('JS5-3', base_demand=0.0, elevation=10, coordinates=(0, -55))
    wn.add_junction('JS5-4', base_demand=0.0, elevation=10, coordinates=(5, -55))
    wn.add_junction('JS5-5', base_demand=0.0, elevation=10, coordinates=(0, -60))

    wn.add_pipe('PS5-1', 'JS5-1', 'JS5-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5-2', 'JS5-1', 'JS5-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5-3', 'JS5-1', 'JS5-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5-4', 'JS5-3', 'JS5-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5-5', 'JS5-2', 'JS5-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS5-6', 'JS5-5', 'JS5-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5-7', 'JS5-5', 'JS5-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS5-8', 'JS5-5', 'JS5-4', length=50, diameter=0.3, roughness=100)

    # Section 6 (going down, based on JS6-1)
    wn.add_junction('JS6-2', base_demand=0.0, elevation=10, coordinates=(-55, -55))
    wn.add_junction('JS6-3', base_demand=0.0, elevation=10, coordinates=(-50, -55))
    wn.add_junction('JS6-4', base_demand=0.0, elevation=10, coordinates=(-45, -55))
    wn.add_junction('JS6-5', base_demand=0.0, elevation=10, coordinates=(-50, -60))

    wn.add_pipe('PS6-1', 'JS6-1', 'JS6-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6-2', 'JS6-1', 'JS6-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6-3', 'JS6-1', 'JS6-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6-4', 'JS6-3', 'JS6-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6-5', 'JS6-2', 'JS6-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS6-6', 'JS6-5', 'JS6-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6-7', 'JS6-5', 'JS6-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS6-8', 'JS6-5', 'JS6-4', length=50, diameter=0.3, roughness=100)

    # Section 7 (going left, based on JS7-1)
    wn.add_junction('JS7-2', base_demand=0.0, elevation=10, coordinates=(-55, -5))
    wn.add_junction('JS7-3', base_demand=0.0, elevation=10, coordinates=(-55, 0))
    wn.add_junction('JS7-4', base_demand=0.0, elevation=10, coordinates=(-55, 5))
    wn.add_junction('JS7-5', base_demand=0.0, elevation=10, coordinates=(-60, 0))

    wn.add_pipe('PS7-1', 'JS7-1', 'JS7-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7-2', 'JS7-1', 'JS7-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7-3', 'JS7-1', 'JS7-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7-4', 'JS7-3', 'JS7-4', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7-5', 'JS7-2', 'JS7-3', length=50, diameter=0.3, roughness=100)

    wn.add_pipe('PS7-6', 'JS7-5', 'JS7-2', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7-7', 'JS7-5', 'JS7-3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('PS7-8', 'JS7-5', 'JS7-4', length=50, diameter=0.3, roughness=100)


    # Save network file (optional)
    #wn.write_inpfile('custom_wdn.inp')

    # You can simulate or visualize now:
    # wntr.graphics.plot_network(wn)
    return wn


def main():

    one_day_in_seconds = 86400
    global_timestep = 300
    wn = custom_complex_water_model()

    wn.add_pattern('gauss_pattern_1', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([1, 2, 4, 7, 10, 7, 4, 2, 1, 0.5], global_timestep, simulation_duration=one_day_in_seconds))
    wn.add_pattern('gauss_pattern_2', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([1, 0.5, 1, 2, 4, 7, 10, 7, 4, 2], global_timestep, simulation_duration=one_day_in_seconds))
    wn.add_pattern('gauss_pattern_3', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([4, 2, 1, 0.5, 1, 2, 4, 7, 10, 7], global_timestep, simulation_duration=one_day_in_seconds))
    wn.add_pattern('gauss_pattern_4', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([10, 7, 4, 2, 1, 0.5, 1, 2, 4, 7], global_timestep, simulation_duration=one_day_in_seconds))
    wn.add_pattern('gauss_pattern_5', InteractiveWNTRSimulator.expand_pattern_to_simulation_duration([7, 10, 7, 4, 2, 1, 0.5, 1, 2, 4], global_timestep, simulation_duration=one_day_in_seconds))

    sim = InteractiveWNTRSimulator(wn)
    sim.init_simulation(duration=one_day_in_seconds, global_timestep=global_timestep) 

    start = time.time()

    sims = [sim]

    node_list = wn.junction_name_list
    link_list = wn.link_name_list

    sim.add_demand('JS1-3', 1, name='gauss_pattern_1')
    sim.add_demand('JS2-3', 1, name='gauss_pattern_2')
    sim.add_demand('JS3-3', 1, name='gauss_pattern_3')
    sim.add_demand('JS4-3', 1, name='gauss_pattern_4')
    sim.add_demand('JS5-3', 1, name='gauss_pattern_5')
    sim.add_demand('JS6-3', 1, name='gauss_pattern_1')
    sim.add_demand('JS7-3', 1, name='gauss_pattern_2')
    sim.add_demand('JS8-3', 1, name='gauss_pattern_3')

    try:
        while not sim.is_terminated():
            #print(f"Current time: {current_time} {current_time / sim.hydraulic_timestep()}")
            current_time = sim.get_sim_time()
            

            sim.step_sim()

        end = time.time()
        
        
        if sim.get_sim_time() >= one_day_in_seconds - global_timestep:
            print(f"Simulation completed in {end - start} seconds.")
            sim.plot_network_over_time(node_key='satisfied_demand', link_key='flowrate', node_labels=True, link_labels=False)
        else:
            print("Simulation terminated before reaching the end time.")

    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            print("Simulation interrupted by user.")
            sys.exit()
        else:
            print("An error occurred during the simulation.")


def agent_rl_main():
    hidraulic_env = WDNEnv(simulation_file_path="nodes_and_inps/L-TOWN_Real.inp", global_timestep=15, duration=86400)


#main()
random_simulation_test()
#agent_rl_main()

'''
    #wn = wntr.network.WaterNetworkModel("Modelli Belmonte Castel Sant'Angelo_2024-09-05_0850/Modello_Castel'Sant'Angelo/_Modelli idraulici/Pacchetto_CSA/CSA_Base.inp")
    #with open('nodes_coords_epanet.csv', 'w') as f:
    #    f.write('Node,Latitude,Longitude\n')
    #    for name, node in wn.nodes.items():
    #        lat, lon = node.coordinates
    #        f.write(f'{name},{lat},{lon}\n')
    #    wn = wntr.network.WaterNetworkModel('NET_4.inp')
     #if sim.get_sim_time() == sim.hydraulic_timestep() * 1:
    #    #sim.start_leak('H1', 0.1, name='ptn_1')
    #    sim.change_demand('H1', 1, name='ptn_1')
    #    #sim.check_mass_balance()
    #
    #if sim.get_sim_time() == sim.hydraulic_timestep() * 60:
    #    sim.set_tank_head('R1', 300)
    #
    #
    #if sim.get_sim_time() == sim.hydraulic_timestep() * 90:
    #    sim.set_tank_head('R1', 2000)

    #if sim.hydraulic_timestep() * 5 <= current_time <= sim.hydraulic_timestep() * 35: 
    #    print(f"Diameter: { sim._wn.get_link('PR8').diameter }")
    #    print(f"Diameter: { sim._wn.get_link('PR8').flow }")
    #if results.node.get('pressure') is not None:
    #    print(len(results.node['pressure']['J8']))

    #if current_time % (sim.hydraulic_timestep() * 3) == 0 and index < len(args):
    #    name, demand, pattern = args[index]
    #    sim.change_demand(name, demand, name=pattern)
    #    index += 1
    #    sim.set_pump_speed('P1', 200.0)

    #if current_time == sim.hydraulic_timestep() * 1:
    #    sim.change_demand('22', 1.1, name='house1_pattern')
    #    sim.change_demand('14', 1, name='house1_pattern')
    #
    #
    #if current_time == sim.hydraulic_timestep() * 25:
    #    sim.start_leak('16', 0.01)
    #
    #if current_time == sim.hydraulic_timestep() * 35:
    #    sim.change_demand('14', name='house1_pattern')
    #
    #if current_time == sim.hydraulic_timestep() * 75:
    #    sim.stop_leak('16')












    #if current_time == sim.hydraulic_timestep() * 50:
    #    sim.change_demand('H2', 3.0)
    

        
    #if current_time == sim.hydraulic_timestep() * 40:
    #    sim.set_pump_head_curve('P1', [(0, 0)])
    #if current_time == sim.hydraulic_timestep() * 40:
    #    sim.set_pump_power('P1', 200.0)
    #    #sim.set_pump_head_curve('P1', 'Pump1_Curve')
    #
    #
    #if current_time == sim.hydraulic_timestep() * 180:
    #    sim.close_pump('P1')
    
    #if current_time == sim.hydraulic_timestep() * 90:
    #    sim.end_outage()
    
    #if current_time == sim.hydraulic_timestep() * 17:
    #    sim.start_leak('J8')
    #
    #if current_time == sim.hydraulic_timestep() * 33:
    #    sim.change_demand('H1', 1, name='house1_pattern')
    #
    #if current_time == sim.hydraulic_timestep() * 48:
    #    branched_sim_1 = sim.branch()
    #    sims.append(branched_sim_1)
    #
    #if current_time == sim.hydraulic_timestep() * 57:
    #    branched_sim_1.change_demand('H1', 1, name='house1_pattern')
    #    sim.stop_leak('J8')
    #
    #if current_time == sim.hydraulic_timestep() * 77:
    #   branched_sim_1.stop_leak('J8')

    #if current_time == sim.hydraulic_timestep() * 0:
    #   #sim.toggle_demand('H1', 0.1, name='house1_pattern')
    #   sim.start_leak('R1', 0.1)
    
    
    #elif cu18rrent_time == sim.hydraulic_timestep() * 26:
    #    sim.toggle_demand('H1', 0.2, name='house2_pattern')
    #
    #elif current_time == sim.hydraulic_timestep() * 37:
    #    sim.stop_leak('J1')
    #
    #if current_time == sim.hydraulic_timestep() * 145:
    #    b = time.time()
    #    sim.extract_snapshot(filename='snapshot_2.json')
    #    e = time.time()
    #    print(f"Elapsed time: {e - b}")
    #    #print(sim.extract_snapshot(filename=None))
    #
    #
    #    sim.toggle_demand('H1', name='house1_pattern')
    #    sim.toggle_demand('H1', name='house2_pattern')
    #    sim.toggle_demand('H1', name='house3_pattern')
    #
    #elif current_time == sim.hydraulic_timestep() * 178:
    #    sim.close_pump('P1')
    #
    #elif current_time == sim.hydraulic_timestep() * 211:
    #    sim.open_pump('P1')

    #elif current_time == sim.hydraulic_timestep() * 60:
    #    #branched_sim_1 = sim.branch()
    #    #branched_sim_2 = sim.branch()
    #    #sims.append(branched_sim_1)
    #    #sims.append(branched_sim_2)
    #    #branched_sim_1.start_leak('J1', 0.1)
    #
    #
    #elif current_time == sim.hydraulic_timestep() * 100:
    #    branched_sim_1.stop_leak('J1')
    #    branched_sim_2.close_pipe('PR0')
    #    branched_sim_2.close_pipe('PR1')
    #    branched_sim_2.close_valve('V1')
    #
    #    
    #
    #elif current_time == sim.hydraulic_timestep() * 80:
    #    branched_sim_2.stop_leak('J1')pressure
'''
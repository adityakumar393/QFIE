import numpy as np
import skfuzzy as fuzz
from QFIE.FuzzyEngines import QuantumFuzzyEngine

# --- 1. Define Universes and Fuzzy Sets ---
# Define the ranges for each variable
env_light = np.linspace(120, 220, 200)
changing_rate = np.linspace(-10, 10, 200)
dimmer_control = np.linspace(0, 10, 200)

# Define the fuzzy membership functions for 'env_light'
l_dark = fuzz.trapmf(env_light, [120, 120, 130, 150])
l_medium = fuzz.trapmf(env_light, [130, 150, 190, 210])
l_light = fuzz.trapmf(env_light, [190, 210, 220, 220])

# Define fuzzy sets for 'changing_rate'
r_ns = fuzz.trimf(changing_rate, [-10, -10, 0])
r_zero = fuzz.trimf(changing_rate, [-10, 0, 10])
r_ps = fuzz.trimf(changing_rate, [0, 10, 10])

# Define fuzzy sets for the output 'dimmer_control'
dm_vs = fuzz.trapmf(dimmer_control, [0, 0, 2, 4])
dm_s = fuzz.trimf(dimmer_control, [2, 4, 6])
dm_b = fuzz.trimf(dimmer_control, [4, 6, 8])
dm_vb = fuzz.trapmf(dimmer_control, [6, 8, 10, 10])

# --- 2. Define the Fuzzy Rules ---
rules = [
    'if env_light is dark and change_rate is neg_small then dimmer_ctrl is very_big',
    'if env_light is medium and change_rate is pos_small then dimmer_ctrl is small',
    'if env_light is light and change_rate is zero then dimmer_ctrl is small',
    'if env_light is light and change_rate is neg_small then dimmer_ctrl is big'
]

# --- 3. Initialize and Configure the Quantum Fuzzy Engine ---
# We use 'linear' encoding which is easier to understand.
qfie = QuantumFuzzyEngine(verbose=True, encoding='linear')

# Add the input variables
qfie.input_variable(name='env_light', range=env_light)
qfie.input_variable(name='change_rate', range=changing_rate)

# Add the output variable
qfie.output_variable(name='dimmer_ctrl', range=dimmer_control)

# Add the fuzzy sets to each variable
qfie.add_input_fuzzysets(var_name='env_light', set_names=['dark', 'medium', 'light'], sets=[l_dark, l_medium, l_light])
qfie.add_input_fuzzysets(var_name='change_rate', set_names=['neg_small', 'zero', 'pos_small'], sets=[r_ns, r_zero, r_ps])
qfie.add_output_fuzzysets(var_name='dimmer_ctrl', set_names=['very_small', 'small', 'big', 'very_big'], sets=[dm_vs, dm_s, dm_b, dm_vb])

# Set the rules
qfie.set_rules(rules)

# --- 4. Build the Quantum Circuit ---
# Provide the crisp input values here. Let's say light is 170 and the rate is 0.
crisp_inputs = {'env_light': 170, 'change_rate': 0}
# 'distributed=False' creates a single quantum circuit for the whole system.
qfie.build_inference_qc(crisp_inputs, draw_qc=True, distributed=False)

# --- 5. Execute and Get the Result ---
# Run the circuit on the simulator for 1000 shots.
crisp_output, activation_values = qfie.execute(n_shots=1000, plot_histo=True)

print(f"\nCrisp Input: {crisp_inputs}")
print(f"Crisp Output (Dimmer Level): {crisp_output}")

import os
import simulation

models = [
    "flat_NoDecay",
    "flat_Decay",
    "hybrid_FreeWeights_NoDecay",
    "hybrid_FreeWeights_Decay",
    "hybrid_Pearce_Hall_NoDecay",
    "hybrid_Pearce_Hall_Decay",
    "full_Bayesian"
]

array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
array_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

# list of all model x parameter-set combinations
jobs = []

for model_type in models:
    n_parameter_sets = len(simulation.get_parameter_combinations(model_type))

    for parameter_set in range(1, n_parameter_sets + 1):
        jobs.append((model_type, parameter_set))

# give this array task only its own subset
assigned_jobs = jobs[array_id::array_count]

for model_type in models:

    parameter_set_ids = [
        parameter_set
        for job_model, parameter_set in assigned_jobs
        if job_model == model_type
    ]

    if parameter_set_ids:
        simulation.run_model_simulation(
            model_type = model_type,
            seed = 1,
            output_dir = "/data/42-julia-hpc-rz-neurowis/shx98ds/compass_model/simulation_results",
            parameter_set_ids = parameter_set_ids
        )

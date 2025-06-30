# carbon_emission_layer.py
from carbontracker.tracker import CarbonTracker

def train_with_carbontracker(train_func, max_epochs, *args, **kwargs):
    tracker = CarbonTracker(epochs=max_epochs)
    results = None
    for epoch in range(max_epochs):
        tracker.epoch_start()
        results = train_func(epoch, *args, **kwargs)
        tracker.epoch_end()
    tracker.stop()
    return results

def parse_carbontracker_log(log_file='carbontracker.log'):
    # Simple parser to extract last run's emissions from the log file
    emissions = None
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "CO2eq:" in line:
                    emissions = line.strip()
                    break
    except Exception:
        emissions = "No emissions data found."
    return emissions

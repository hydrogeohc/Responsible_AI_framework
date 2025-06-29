import flwr as fl

if __name__ == "__main__":
    # Default FedAvg strategy; customize as needed
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(server_address="localhost:8080", strategy=strategy)

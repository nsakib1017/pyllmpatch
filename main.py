def main() -> None:
    from pipeline.config import load_runtime_config
    from pipeline.runner import run_experiment

    run_experiment(load_runtime_config())


if __name__ == "__main__":
    main()

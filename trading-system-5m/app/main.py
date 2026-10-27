from app.runner import TradingSystemRunner


def main():
    runner = TradingSystemRunner()
    runner.run_forever()


if __name__ == "__main__":
    main()

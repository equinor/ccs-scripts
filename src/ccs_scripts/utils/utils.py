import logging
import time


class Timer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._timings = {}
            cls.code_parts = {"total": "Total"}
        return cls._instance

    def start(self, name: str):
        if name not in self._timings:
            self._timings[name] = {"start": None, "elapsed": 0}
        self._timings[name]["start"] = time.time()

    def stop(self, name):
        if name in self._timings and self._timings[name]["start"] is not None:
            elapsed_time = time.time() - self._timings[name]["start"]
            self._timings[name]["elapsed"] += elapsed_time
            self._timings[name]["start"] = None

    def get_elapsed_time(self, name):
        return self._timings.get(name, {}).get("elapsed", 0)

    def report(self):
        max_len_category = max([len(x) for x in self.code_parts.values()])
        print(max_len_category)
        logging.info("\nPerformance Timing Report:")
        logging.info(f"\n{'Category':<{max_len_category + 1}}  {'Time (s)':>10}")
        logging.info("=" * (max_len_category + 13))
        for name, timing in self._timings.items():
            if name == "total":
                continue
            desc = self.code_parts[name] if name in self.code_parts else name
            logging.info(f"{desc:<{max_len_category + 1}}: {timing['elapsed']:>10.2f}")
        if "total" in self._timings:
            misc = self._timings["total"]["elapsed"] - sum(
                [y["elapsed"] for x, y in self._timings.items() if x != "total"]
            )
            logging.info(f"{'Miscellaneous':<{max_len_category + 1}}: {misc:>10.2f}")
            logging.info("-" * (max_len_category + 13))
            logging.info(
                f"{'Total':<{max_len_category + 1}}: {self._timings['total']['elapsed']:>10.2f}"
            )
        logging.info("")

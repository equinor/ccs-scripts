import logging
import time
from typing import Optional


class Timer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._timings = {}
            cls.code_parts = {"total": "Total"}
        return cls._instance

    def reset_timings(self):
        self._timings = {}

    def start(self, name: str, parent: Optional[str] = None):
        active = [
            x
            for x, y in self._timings.items()
            if y["start"] is not None and x not in ["total", parent]
        ]
        if len(active) > 0:
            # For debugging:
            # print(
            #     f"\nTrying to start timings on {name}, "
            #     f"but some timings are already active: {active}"
            # )
            # exit()
            return
        if name not in self._timings:
            self._timings[name] = {"start": None, "elapsed": 0, "parent": parent}
        self._timings[name]["start"] = time.time()

    def stop(self, name: str):
        if name in self._timings and self._timings[name]["start"] is not None:
            elapsed_time = time.time() - self._timings[name]["start"]
            self._timings[name]["elapsed"] += elapsed_time
            self._timings[name]["start"] = None

    def report(self):
        max_len_category = 4 + max([len(x) for x in self.code_parts.values()])
        logging.info("\nPerformance Timing Report:")
        logging.info(f"\n{'Category':<{max_len_category + 1}}  {'Time (s)':>10}")
        logging.info("=" * (max_len_category + 13))
        for name, timing in self._timings.items():
            if name in ["total", "logging"] or timing["parent"] is not None:
                continue
            desc = self.code_parts[name] if name in self.code_parts else name
            logging.info(f"{desc:<{max_len_category + 1}}: {timing['elapsed']:>10.2f}")
            # Find childs:
            childs_found = False
            for name2, timing2 in self._timings.items():
                if timing2["parent"] == name:
                    childs_found = True
                    desc = self.code_parts[name2] if name2 in self.code_parts else name2
                    logging.info(
                        f"  - {desc:<{max_len_category - 3}}: "
                        f"{timing2['elapsed']:>10.2f}"
                    )
            if childs_found:
                misc = timing["elapsed"] - sum(
                    [
                        y["elapsed"]
                        for x, y in self._timings.items()
                        if y["parent"] == name
                    ]
                )
                logging.info(
                    f"{'  - Miscellaneous':<{max_len_category + 1}}: {misc:>10.2f}"
                )
        if "logging" in self._timings:
            timing = self._timings["logging"]
            logging.info(
                f"{'Various logging':<{max_len_category + 1}}: "
                f"{timing['elapsed']:>10.2f}"
            )
        if "total" in self._timings:
            misc = self._timings["total"]["elapsed"] - sum(
                [
                    y["elapsed"]
                    for x, y in self._timings.items()
                    if x != "total" and y["parent"] is None
                ]
            )
            logging.info(f"{'Miscellaneous':<{max_len_category + 1}}: {misc:>10.2f}")
            logging.info("-" * (max_len_category + 13))
            logging.info(
                f"{'Total':<{max_len_category + 1}}: "
                f"{self._timings['total']['elapsed']:>10.2f}"
            )
        logging.info("")

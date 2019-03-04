import unittest

from circuit import LocalContext, make_circuit


class TestCircuit(unittest.TestCase):
    def run_evaluator(self):
        with LocalContext():
            evaluator = make_circuit('ThreeStageTranimpedenceAmplifier').evaluator()
            #         return {'W1': 2, 'WL1': 2, 'WB1': 2, 'W2': 2, 'WL2': 2, 'WB2': 2, 'W3': 2, 'WB3': 2, 'WB': 2,
            #                 'L1': 1, 'LL1': 1, 'LB1': 1, 'L2': 1, 'LL2': 1, 'LB2': 1, 'L3': 1, 'LB3': 1, 'LB': 1,
            #                 'RB': 10}
            # simulator = ThreeSimu2(wmax=1000, lmax=100, rmax=1000, base_tmp_path='./tmp', suppress_exception_reward=-5)

            evaluator.set_bound('W1', 2, 1000)
            evaluator.set_bound('WL1', 2, 1000)
            evaluator.set_bound('WB1', 2, 1000)

            evaluator.set_bound('W2', 2, 1000)
            evaluator.set_bound('WL2', 2, 1000)
            evaluator.set_bound('WB2', 2, 1000)

            evaluator.set_bound('W3', 2, 1000)
            evaluator.set_bound('WB3', 2, 1000)
            evaluator.set_bound('WB', 2, 1000)

            evaluator.set_bound('L1', 1, 100)
            evaluator.set_bound('LL1', 1, 100)
            evaluator.set_bound('LB1', 1, 100)

            evaluator.set_bound('L2', 1, 100)
            evaluator.set_bound('LL2', 1, 100)
            evaluator.set_bound('LB2', 1, 100)

            evaluator.set_bound('L3', 1, 100)
            evaluator.set_bound('LB3', 1, 100)
            evaluator.set_bound('LB', 1, 100)

            evaluator.set_bound('RB', 10, 1000)

            print(evaluator.sample())

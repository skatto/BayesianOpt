
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


def _to_dict(p_list, param):
    return {p['name']: p['dtype'](v) for p, v in zip(p_list, param)}


class BayesianOpt(object):
    '''
    First, please call add_param().
    and call setup()

    example:
        bo = BayesianOpt()
        bo.add_param('param1', vmin=1, vmax=10, step=1, dtype=np.int)
        bo.add_param('param2', vmin=0, vmax=1, step=0.1, dtype=np.float)
        bo.setup()

        def f(param1, param2):
                return param1 * np.sin(param2)

        for i in range(10):
            p = bo.get_next_param()
            bo.add_result(p, f(**p))
        print(*bo.get_optimized_one())
    '''

    def __init__(self):
        self.params = list()
        self._is_setup = False

        # init in setup()
        self.grid = None
        self.X = None
        self.Y = None
        self.done = None

    def add_param(self, name, *, vmin, vmax, step, dtype=np.float):
        '''
        Args:
            vmin, vmax  : value max and min
            step  : value step
            dtype  : type of param, numpy.float or numpy.int
        '''
        if vmax <= vmin:
            raise ValueError(f'Must be vmax > vmin : vmax={vmax}, vmin={vmin}')

        if vmax - vmin <= step:
            raise ValueError('Must be # of param > 1 :'
                             f'vmax-vmin={vmax-vmin}, step={step}')

        if dtype not in [np.float, np.int]:
            raise ValueError('dtype must be numpy.float or numpy.int,'
                             f'but not {dtype}')

        if self._is_setup:
            raise RuntimeError('Call this function, Before Calling setup()')

        grid = np.arange(vmin, vmax + step, step, dtype)
        self.params.append({'name': name, 'grid': grid, 'dtype': dtype})

    def setup(self):
        if len(self.params) == 0:
            raise RuntimeError('Call setup() after calling add_param()')

        if self._is_setup:
            raise Warning('Already setup')
            return

        self.grid = [[]]
        for param in self.params:
            buf = []
            for new_x in param['grid']:
                buf.extend([x + [new_x] for x in self.grid])
            self.grid = buf

        self.grid = np.asarray(self.grid)
        self.X = []
        self.Y = []
        self.done = np.array([False] * len(self.grid), dtype=np.bool)

        self._is_setup = True

    def get_next_param(self, aggressiveness=2):
        ''' return dict of parameter '''
        if not self._is_setup:
            raise RuntimeError('Call this function, After Calling setup()')

        if len(self.X) == 0:
            next_param = self.grid[np.random.choice(len(self.grid))]
        else:
            gp = GaussianProcessRegressor()
            gp.fit(np.asarray(self.X), np.asarray(self.Y))
            mean, sigma = gp.predict(self.grid, return_std=True)

            masked = np.ma.array(mean + sigma * aggressiveness, mask=self.done)
            next_param = self.grid[np.argmax(masked)]

        return _to_dict(self.params, next_param)

    def get_next_params(self, n, aggressiveness=2):
        ''' return array of dict of parameter '''
        if not self._is_setup:
            raise RuntimeError('Call this function, After Calling setup()')

        if len(self.X) == 0:
            next_param = self.grid[np.random.choice(len(self.grid))]
        else:
            gp = GaussianProcessRegressor()
            gp.fit(np.asarray(self.X), np.asarray(self.Y))
            mean, sigma = gp.predict(self.grid, return_std=True)

            masked = np.ma.array(mean + sigma * aggressiveness, mask=self.done)
            next_param = self.grid[np.argmax(masked)]

        return _to_dict(self.params, next_param)

    def add_result(self, param, result):
        if not self._is_setup:
            raise RuntimeError('Call this function, After Calling setup()')

        self.X.append([param[p['name']] for p in self.params])
        self.Y.append(result)

        for i, p in enumerate(self.grid):
            if np.allclose(p, self.X[-1]):
                self.done[i] = True
                break

    def get_optimized_one(self):
        ''' return optimized parameter, and optimized result '''
        if not self._is_setup:
            raise RuntimeError('Call this function, After Calling setup()')

        idx = np.argmax(self.Y)
        return _to_dict(self.params, self.grid[idx]), self.Y[idx]

import sys
import time

import numpy as np
from torch.autograd import Variable


def iwae_eval(model, loader, n_sample=100):
    model.eval()
    time_ = time.time()
    elbos = []
    for i, (batch, _) in enumerate(loader):
        batch = Variable(batch).type(model.dtype)
        elbo, logpx, logpz, logqz = model(batch, k=n_sample)
        elbos.append(elbo.data[0])

        sys.stderr.write("batch %d, stats %.4f\n" % (i, elbo.data[0]))

    mean_ = np.mean(elbos)
    print(mean_, 'T:', time.time() - time_)
    return mean_

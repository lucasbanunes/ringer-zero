# Based on: https://github.com/ringer-softwares/neuralnet
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
import numpy as np
import collections
import pandas

from . import logger


def sp_func(pd, fa):
    return np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))


#
# Decorate the history dictionary after the training phase with some useful controll values
#
class Summary:

    #
    # Constructor
    #
    def __init__(self, detailed=False):
        self.detailed = detailed


#
# Use this method to decorate the keras history in the end of the training
#
def __call__(self, history, kw):

    d = {}
    x_train, y_train = kw["data"]
    x_val, y_val = kw["data_val"]
    model = kw["model"]

    # Get the number of events for each set (train/val). Can be used to approx the number of
    # passed events in pd/fa analysis. Use this to integrate values (approx)
    sgn_total = len(y_train[y_train == 1])
    bkg_total = len(y_train[y_train != 1])
    sgn_total_val = len(y_val[y_val == 1])
    bkg_total_val = len(y_val[y_val != 1])

    logger.info("Starting the train summary...")

    y_pred = model.predict(x_train, batch_size=1024, verbose=0)
    y_pred_val = model.predict(x_val, batch_size=1024, verbose=0)

    # get vectors for op mode (train+val)
    y_pred_op = np.concatenate((y_pred, y_pred_val), axis=0)
    y_op = np.concatenate((y_train, y_val), axis=0)

    # No threshold is needed
    d['auc'] = roc_auc_score(y_train, y_pred)
    d['auc_val'] = roc_auc_score(y_val, y_pred_val)
    d['auc_op'] = roc_auc_score(y_op, y_pred_op)

    # No threshold is needed
    d['mse'] = mean_squared_error(y_train, y_pred)
    d['mse_val'] = mean_squared_error(y_val, y_pred_val)
    d['mse_op'] = mean_squared_error(y_op, y_pred_op)

    if self.detailed:
        d['rocs'] = {}
        d['hists'] = {}

    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_train, y_pred)
    sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    step = 1e-2
    bins = np.arange(min(y_train), max(y_train)+step, step=step)

    if self.detailed:
        d['rocs']['roc'] = (pd, fa)
        d['hists']['trn_sgn'] = np.histogram(y_pred[y_train == 1], bins=bins)
        d['hists']['trn_bkg'] = np.histogram(y_pred[y_train != 1], bins=bins)

    logger.info("Train samples     : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
                pd[knee], fa[knee], sp[knee], d['auc'], d['mse'])

    d['max_sp_pd'] = (pd[knee], int(pd[knee]*sgn_total), sgn_total)
    d['max_sp_fa'] = (fa[knee], int(fa[knee]*bkg_total), bkg_total)
    d['max_sp'] = sp[knee]
    d['acc'] = accuracy_score(y_train, y_pred > threshold)

    # Validation
    fa, pd, thresholds = roc_curve(y_val, y_pred_val)
    sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    if self.detailed:
        d['rocs']['roc_val'] = (pd, fa)
        d['hists']['val_sgn'] = np.histogram(y_pred_val[y_val == 1], bins=bins)
        d['hists']['val_bkg'] = np.histogram(y_pred_val[y_val != 1], bins=bins)

    logger.info("Validation Samples: Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
                pd[knee], fa[knee], sp[knee], d['auc_val'], d['mse_val'])

    d['max_sp_pd_val'] = (pd[knee], int(pd[knee]*sgn_total_val), sgn_total_val)
    d['max_sp_fa_val'] = (fa[knee], int(fa[knee]*bkg_total_val), bkg_total_val)
    d['max_sp_val'] = sp[knee]
    d['acc_val'] = accuracy_score(y_val, y_pred_val > threshold)

    # op
    fa, pd, thresholds = roc_curve(y_op, y_pred_op)
    sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    if self.detailed:
        d['rocs']['roc_op'] = (pd, fa)
        d['hists']['op_sgn'] = np.histogram(y_pred_op[y_op == 1], bins=bins)
        d['hists']['op_bkg'] = np.histogram(y_pred_op[y_op != 1], bins=bins)

    logger.info("op Samples : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
                pd[knee], fa[knee], sp[knee], d['auc_val'], d['mse_val'])

    d['threshold_op'] = threshold
    d['max_sp_pd_op'] = (pd[knee], int(
        pd[knee]*(sgn_total+sgn_total_val)), (sgn_total+sgn_total_val))
    d['max_sp_fa_op'] = (fa[knee], int(
        fa[knee]*(bkg_total+bkg_total_val)), (bkg_total+bkg_total_val))
    d['max_sp_op'] = sp[knee]
    d['acc_op'] = accuracy_score(y_op, y_pred_op > threshold)

    history['summary'] = d


#
# Use this class to decorate the history with the reference values configured by the user
#
class Reference:

    #
    # Constructor
    #
    def __init__(self, refs):
        self.__references = collections.OrderedDict()

        for key, ref in refs.items():

            pd = [ref['det']['passed'], ref['det']['total']]
            pd = [pd[0]/pd[1], pd[0], pd[1]]
            fa = [ref['fake']['passed'], ref['fake']['total']]
            fa = [fa[0]/fa[1], fa[0], fa[1]]
            logger.info('%s (pd=%1.2f, fa=%1.2f, sp=%1.2f)', key,
                        pd[0]*100, fa[0]*100, sp_func(pd[0], fa[0])*100)
            self.__references[key] = {
                'pd': pd, 'fa': fa, 'sp': sp_func(pd[0], fa[0])}

    #
    # decorate the history after the training phase
    #
    def __call__(self, history, kw):
        model = kw["model"]
        x_train, y_train = kw["data"]
        x_val, y_val = kw["data_val"]

        y_pred = model.predict(x_train, batch_size=1024, verbose=0)
        y_pred_val = model.predict(x_val, batch_size=1024, verbose=0)

        # get vectors for operation mode (train+val)
        y_pred_operation = np.concatenate((y_pred, y_pred_val), axis=0)
        y_operation = np.concatenate((y_train, y_val), axis=0)

        # train_total = len(y_train)
        # val_total = len(y_val)

        # Here, the threshold is variable and the best values will
        # be setted by the max sp value found in hte roc curve
        # Training
        fa, pd, thresholds = roc_curve(y_train, y_pred)
        sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))

        # Validation
        fa_val, pd_val, thresholds_val = roc_curve(y_val, y_pred_val)
        sp_val = np.sqrt(np.sqrt(pd_val*(1-fa_val))
                         * (0.5*(pd_val+(1-fa_val))))

        # Operation
        fa_op, pd_op, thresholds_op = roc_curve(y_operation, y_pred_operation)
        sp_op = np.sqrt(np.sqrt(pd_op*(1-fa_op)) * (0.5*(pd_op+(1-fa_op))))

        history['reference'] = {}

        for key, ref in self.__references.items():
            d = self.calculate(y_train, y_val, y_operation, ref, pd, fa, sp, thresholds,
                               pd_val, fa_val, sp_val, thresholds_val, pd_op, fa_op, sp_op, thresholds_op)
            logger.info("          : %s", key)
            logger.info("Reference : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ",
                        ref['pd'][0]*100, ref['fa'][0]*100, ref['sp']*100)
            logger.info("Train     : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ",
                        d['pd'][0]*100, d['fa'][0]*100, d['sp']*100)
            logger.info("Validation: [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ",
                        d['pd_val'][0]*100, d['fa_val'][0]*100, d['sp_val']*100)
            logger.info("Operation : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ",
                        d['pd_op'][0]*100, d['fa_op'][0]*100, d['sp_op']*100)
            history['reference'][key] = d

    #
    # Calculate sp, pd and fake given a reference
    #
    def calculate(self, y_train, y_val, y_op, ref, pd, fa, sp, thresholds, pd_val, fa_val, sp_val, thresholds_val, pd_op, fa_op, sp_op, thresholds_op):

        d = {}

        def closest(values, ref):
            index = np.abs(values-ref)
            index = index.argmin()
            return values[index], index

        # Check the reference counts
        op_total = len(y_op[y_op == 1])
        if ref['pd'][2] != op_total:
            ref['pd'][2] = op_total
            ref['pd'][1] = int(ref['pd'][0]*op_total)

        # Check the reference counts
        op_total = len(y_op[y_op != 1])
        if ref['fa'][2] != op_total:
            ref['fa'][2] = op_total
            ref['fa'][1] = int(ref['fa'][0]*op_total)

        d['pd_ref'] = ref['pd']
        d['fa_ref'] = ref['fa']
        d['sp_ref'] = ref['sp']
        # d['reference'] = ref['reference']

        # Train
        _, index = closest(pd, ref['pd'][0])
        train_total = len(y_train[y_train == 1])
        d['pd'] = (pd[index],  int(train_total*float(pd[index])), train_total)
        train_total = len(y_train[y_train != 1])
        d['fa'] = (fa[index],  int(train_total*float(fa[index])), train_total)
        d['sp'] = sp_func(d['pd'][0], d['fa'][0])
        d['threshold'] = thresholds[index]

        # Validation
        _, index = closest(pd_val, ref['pd'][0])
        val_total = len(y_val[y_val == 1])
        d['pd_val'] = (pd_val[index],  int(
            val_total*float(pd_val[index])), val_total)
        val_total = len(y_val[y_val != 1])
        d['fa_val'] = (fa_val[index],  int(
            val_total*float(fa_val[index])), val_total)
        d['sp_val'] = sp_func(d['pd_val'][0], d['fa_val'][0])
        d['threshold_val'] = thresholds_val[index]

        # Train + Validation
        _, index = closest(pd_op, ref['pd'][0])
        op_total = len(y_op[y_op == 1])
        d['pd_op'] = (pd_op[index],  int(
            op_total*float(pd_op[index])), op_total)
        op_total = len(y_op[y_op != 1])
        d['fa_op'] = (fa_op[index],  int(
            op_total*float(fa_op[index])), op_total)
        d['sp_op'] = sp_func(d['pd_op'][0], d['fa_op'][0])
        d['threshold_op'] = thresholds_op[index]

        return d


class Evaluator:

    #
    # Constructor
    #
    def __init__(self, eval_file_name: str,
                 filter_query: str = '(target == 1 and el_lhmedium == 1) or (target != 1 and el_lhvloose != 1)',
                 ringer_version: str = 'v0'):

        self.eval_df = pandas.read_hdf(eval_file_name).query(filter_query)
        self.ringer_version = ringer_version
        self.input_cols = ['trig_L2_cl_ring_%d' % i for i in range(100)]
        if self.ringer_version == 'v1':
            # for new training, we selected 1/2 of rings in each layer
            # pre-sample - 8 rings
            # EM1 - 64 rings
            # EM2 - 8 rings
            # EM3 - 8 rings
            # Had1 - 4 rings
            # Had2 - 4 rings
            # Had3 - 4 rings
            prefix = 'trig_L2_cl_ring_%i'

            # rings presmaple
            presample = [prefix % iring for iring in range(8//2)]

            # EM1 list
            sum_rings = 8
            em1 = [prefix % iring for iring in range(sum_rings, sum_rings+(64//2))]

            # EM2 list
            sum_rings = 8+64
            em2 = [prefix % iring for iring in range(sum_rings, sum_rings+(8//2))]

            # EM3 list
            sum_rings = 8+64+8
            em3 = [prefix % iring for iring in range(sum_rings, sum_rings+(8//2))]

            # HAD1 list
            sum_rings = 8+64+8+8
            had1 = [prefix % iring for iring in range(sum_rings, sum_rings+(4//2))]

            # HAD2 list
            sum_rings = 8+64+8+8+4
            had2 = [prefix % iring for iring in range(sum_rings, sum_rings+(4//2))]

            # HAD3 list
            sum_rings = 8+64+8+8+4+4
            had3 = [prefix % iring for iring in range(sum_rings, sum_rings+(4//2))]

            self.input_cols = presample+em1+em2+em3+had1+had2+had3

    #
    # Use this method to decorate the keras history in the end of the training
    #
    def __call__(self, history, kw):
        d = {}
        x_eval, y_eval = self.eval_df[self.input_cols], self.eval_df['target']
        model = kw["model"]

        # Get the number of events for each set (train/val). Can be used to approx the number of
        # passed events in pd/fa analysis. Use this to integrate values (approx)
        sgn_total = len(y_eval[y_eval == 1])
        bkg_total = len(y_eval[y_eval != 1])

        logger.info("Starting the train summary...")

        y_pred = model.predict(x_eval, batch_size=1024, verbose=0)

        # No threshold is needed
        d['auc_eval'] = roc_auc_score(y_eval, y_pred)

        # No threshold is needed
        d['mse_eval'] = mean_squared_error(y_eval, y_pred)

        if self.detailed:
            d['rocs_eval'] = {}
            d['hists_eval'] = {}

        # Here, the threshold is variable and the best values will
        # be setted by the max sp value found in hte roc curve
        # Training
        fa, pd, thresholds = roc_curve(y_eval, y_pred)
        sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
        knee = np.argmax(sp)
        threshold = thresholds[knee]

        step = 1e-2
        bins = np.arange(min(y_eval), max(y_eval)+step, step=step)

        if self.detailed:
            d['rocs_eval']['roc'] = (pd, fa)
            d['hists_eval']['trn_sgn'] = np.histogram(
                y_pred[y_eval == 1], bins=bins)
            d['hists_eval']['trn_bkg'] = np.histogram(
                y_pred[y_eval != 1], bins=bins)

        logger.info("Evaluation samples     : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
                    pd[knee], fa[knee], sp[knee], d['auc_eval'], d['mse_eval'])

        d['max_sp_pd_eval'] = (pd[knee], int(pd[knee]*sgn_total), sgn_total)
        d['max_sp_fa_eval'] = (fa[knee], int(fa[knee]*bkg_total), bkg_total)
        d['max_sp_eval'] = sp[knee]
        d['acc_eval'] = accuracy_score(y_eval, y_pred > threshold)

        history['summary'] = d

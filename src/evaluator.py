# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import time
import torch
import numpy as np
import sympy as sp

from .utils import to_cuda  # , timeout
from .utils import TimeoutError
from .envs.ode import second_index


TOLERANCE_THRESHOLD = 1e-1


logger = getLogger()


def check_fourier_cond_init(env, src, tgt, hyp):

    try:
        nx = src
        dimension, pos = env.parse_int(nx)
        nx = nx[pos:]
        operateur, nx = env.prefix_to_infix(nx[1:])
        cond_init, nx = env.prefix_to_infix(nx[1:])
        if nx[1:] != []:
            logger.info("wrong src")
            return False

        # read tgt
        reg, pos1 = env.parse_int(tgt)
        stab, pos2 = env.parse_int(tgt[pos1 + 1 :])
        tgt = tgt[pos1 + pos2 :]

        # read hyp
        reghyp, pos1 = env.parse_int(hyp)
        stabhyp, pos2 = env.parse_int(hyp[pos1 + 1 :])
        hyp = hyp[pos1 + pos2 :]

        # compare hyp and tgt
        if (
            reghyp != reg or stabhyp != stab
        ):  # First condition on existence and stability
            # logger.error("Incorrect reg or stab")
            return False

        # predict bounds is a subtask, used for training but not for evaluation,
        # hence the comment, uncomment if bounds are to be used at evaluations
        # if env.predict_bounds:

        #     # read tgt
        #     nr_bounds = tgt.count(env.list_separator)
        #     nr_dimension = tgt.count(env.line_separator)
        #     if nr_bounds != dimension or nr_dimension != dimension:
        #         # logger.error("Incorrect form of tgt in read_fourier")
        #         return False
        #     bounds = []
        #     pos = 1
        #     for i in range(dimension):
        #         tgt = tgt[pos + 1:]
        #         if tgt[0] == env.pos_inf:
        #             bda = np.inf
        #             pos = 1
        #         elif tgt[0] == env.neg_inf:
        #             bda = -np.inf
        #             pos = 1
        #         else:
        #             bda, pos = env.parse_float(tgt)
        #         tgt = tgt[pos + 1:]
        #         if tgt[0] == env.pos_inf:
        #             bdb = np.inf
        #             pos = 1
        #         elif tgt[0] == env.neg_inf:
        #             bdb = -np.inf
        #             pos = 1
        #         else:
        #             bdb, pos = env.parse_float(tgt)
        #         bounds.append([bda, bdb])

        #     # read hyp
        #     nr_bounds = hyp.count(env.list_separator)
        #     nr_dimension = hyp.count(env.line_separator)
        #     if nr_bounds != dimension or nr_dimension != dimension:
        #         # logger.error("Incorrect form of hyp in read_fourier")
        #         return False
        #     bounds_hyp = []
        #     pos = 1
        #     for i in range(dimension):
        #         hyp = hyp[pos + 1:]
        #         if hyp[0] == env.pos_inf:
        #             bda = np.inf
        #             pos = 1
        #         elif hyp[0] == env.neg_inf:
        #             bda = -np.inf
        #             pos = 1
        #         else:
        #             bda, pos = env.parse_float(hyp)
        #         hyp = hyp[pos + 1:]
        #         if hyp[0] == env.pos_inf:
        #             bdb = np.inf
        #             pos = 1
        #         elif hyp[0] == env.neg_inf:
        #             bdb = -np.inf
        #             pos = 1
        #         else:
        #             bdb, pos = env.parse_float(hyp)
        #         bounds_hyp.append([bda, bdb])

        #     # compare hyp and tgt
        #     for i in range(len(bounds)):
        # # Second condition on frequency bounds of initial condition
        #         for j in range(len(bounds[i])):
        #             if abs(bounds[i][j]) == np.inf:
        #                 if bounds[i][j] != bounds_hyp[i][j]:
        #                     # logger.error("Incorrect inf bound prediction")
        #                     return False
        #             elif abs(bounds[i][j]) == 0:
        #                 if bounds_hyp[i][j] != 0:
        #                     # logger.error("Incorrect 0 bound prediction")
        #                     return False
        #             else:
        #                 if (bounds[i][j] - bounds_hyp[i][j]) / bounds[i][j] > 0.1:
        #                     # logger.error("Incorrect bound prediction")
        #                     return False

    except Exception as e:
        logger.info(f"Exception {e} in top_test")
        return False

    return True


def idx_to_infix(env, idx, input=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    infix = env.input_to_infix(prefix) if input else env.output_to_infix(prefix)
    return infix


def compare_gramians(env, tgt, hyp, tolerance, norm1=False):
    nr_lines = tgt.count(env.line_separator)
    nr_cols = tgt.count(env.list_separator)
    nr_cols = nr_cols // nr_lines
    # read hypothesis
    h = hyp
    h_gramian = np.zeros((nr_lines, nr_cols), dtype=float)
    for i in range(nr_lines):
        for j in range(nr_cols):
            val, pos = env.parse_float(h)
            if np.isnan(val):
                return False
            if len(h) <= pos or h[pos] != env.list_separator:
                return False
            h_gramian[i][j] = val
            h = h[pos + 1 :]
        if len(h) == 0 or h[0] != env.line_separator:
            return False
        h = h[1:]
    # read target
    t = tgt
    t_gramian = np.zeros((nr_lines, nr_cols), dtype=float)
    for i in range(nr_lines):
        for j in range(nr_cols):
            val, pos = env.parse_float(t)
            t_gramian[i][j] = val
            t = t[pos + 1 :]
        t = t[1:]
    # compare
    if norm1:
        tot = 0
        nb = 0
        for i in range(nr_lines):
            for j in range(nr_cols):
                if t_gramian[i][j] != h_gramian[i][j]:
                    den = h_gramian[i][j] if t_gramian[i][j] == 0 else t_gramian[i][j]
                    delta = abs((t_gramian[i][j] - h_gramian[i][j]) / den)
                    tot += delta
                    nb += 1

        return tot <= tolerance * nb
    else:
        for i in range(nr_lines):
            for j in range(nr_cols):
                if t_gramian[i][j] != h_gramian[i][j]:
                    den = h_gramian[i][j] if t_gramian[i][j] == 0 else t_gramian[i][j]
                    delta = abs((t_gramian[i][j] - h_gramian[i][j]) / den)
                    if delta > tolerance:
                        return False
    return True


def check_gramian(env, src, tgt, hyp):
    # Read src
    try:
        degree, pos = env.parse_int(src)
        nx = src[
            pos:
        ]  # retourne src sans le degree et le séparateur qui va avec si j'ai bien suivi
        system = []
        while len(nx) > 0:
            b, nx = env.prefix_to_infix(nx[1:])
            # convertit en sympy, on en aura besoin de toutes facons
            s = sp.S(b)
            system.append(s)

        # get expected shape of solution (from tgt)
        nr_lines = tgt.count(env.line_separator)
        nr_cols = tgt.count(env.list_separator)
        if nr_cols % nr_lines != 0 or nr_cols // nr_lines != degree:
            logger.error("Incorrect target gramian in check_gramian")
            return False
        nr_cols = nr_cols // nr_lines

        for i in range(degree):
            valA, valB = env.compute_gradient_control(
                system[i], env.eval_point, degree, nr_lines
            )
            if i == 0:
                A = valA
                B = valB
            else:
                A = np.vstack((A, valA))
                B = np.vstack((B, valB))

        A = A / np.linalg.norm(A)
        B = B / np.linalg.norm(A)

        # read hyp, check correct shape
        h = hyp
        K0 = np.zeros((nr_lines, nr_cols))
        for i in range(nr_lines):
            for j in range(nr_cols):
                val, pos = env.parse_float(h)
                if np.isnan(val):
                    return False
                if len(h) <= pos or h[pos] != env.list_separator:
                    return False
                K0[i][j] = val
                h = h[pos + 1 :]
            if len(h) == 0 or h[0] != env.line_separator:
                return False
            h = h[1:]

        V = A + B @ K0
        return max(np.linalg.eigvals(V).real) < 0

    except TimeoutError:
        return False
    except Exception as e:
        logger.info(f"{e} in check_gramian")
        return False


def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV
    src = [env.id2word[wid] for wid in eq["src"]]
    tgt = [env.id2word[wid] for wid in eq["tgt"]]
    hyp = [env.id2word[wid] for wid in eq["hyp"]]

    if eq["task"] == "ode_convergence_speed":
        try:
            tgt, _ = env.parse_float(tgt)
            l1 = len(hyp)
            hyp, l2 = env.parse_float(hyp)
            if hyp == np.nan or l2 != l1:
                is_valid = False
            elif hyp == tgt:
                is_valid = True
            else:
                den = hyp if tgt == 0 else tgt
                is_valid = abs((tgt - hyp) / den) < TOLERANCE_THRESHOLD
        except Exception:
            is_valid = False
            tgt = 0
            hyp = 0

    elif eq["task"] == "ode_control":
        if env.predict_gramian:
            try:
                d, l1 = env.parse_int(hyp)
                t, l2 = env.parse_int(tgt)
                if d == 0 and t == 0 and not env.auxiliary_task:
                    if env.euclidian_metric:
                        is_valid = compare_gramians(
                            env,
                            tgt[l2 + 1 :],
                            hyp[l1 + 1 :],
                            env.gramian_tolerance,
                            env.gramian_norm1,
                        )
                    else:
                        is_valid = check_gramian(env, src, tgt, hyp[l1 + 1 :])
                else:
                    is_valid = d == t
            except Exception:
                is_valid = False
        else:
            try:
                tgt, _ = env.parse_int(tgt)
                l1 = len(hyp)
                hyp, l2 = env.parse_int(hyp)
                if hyp == np.nan or l2 != l1:
                    is_valid = False
                else:
                    is_valid = hyp == tgt
            except Exception:
                is_valid = False
    elif eq["task"] == "fourier_cond_init":
        try:
            is_valid = check_fourier_cond_init(env, src, tgt, hyp)
        except Exception:
            is_valid = False
    else:
        is_valid = hyp == tgt
    # update hypothesis
    eq["src"] = env.input_to_infix(src)
    eq["tgt"] = tgt
    eq["hyp"] = hyp
    eq["is_valid"] = is_valid
    return eq


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores["total"] = sum(self.trainer.EQUATIONS.values())
            scores["unique"] = len(self.trainer.EQUATIONS)
            scores["unique_prop"] = 100.0 * scores["unique"] / scores["total"]
            return scores

        with torch.no_grad():
            # for data_type in ['valid', 'test']:  FC save time
            for data_type in ["valid"]:
                for task in params.tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam_fast(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)

        return scores

    def truncate_at(self, x, xlen):
        pattern = self.env.word2id[self.env.func_separator]
        bs = len(xlen)
        eos = self.env.eos_index
        assert x.shape[1] == bs
        new_seqs = []
        new_lengths = []
        for i in range(bs):
            s = x[: xlen[i], i].tolist()
            assert s[0] == s[-1] == eos
            ns = second_index(s, pattern)
            if ns != len(s):
                s = s[:ns]
                s.append(eos)
            new_seqs.append(s)
            new_lengths.append(len(s))

        # batch sequence
        lengths = torch.LongTensor(new_lengths)
        seqs = torch.LongTensor(lengths.max().item(), bs).fill_(self.env.pad_index)
        for i, s in enumerate(new_seqs):
            seqs[: lengths[i], i].copy_(torch.LongTensor(s))

        return seqs, lengths

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in [
            "ode_convergence_speed",
            "ode_control",
            "fourier_cond_init",
        ]

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # print status
            if n_total.sum().item() % 500 < params.batch_size_eval:
                logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # optionally truncate input
            x1_, len1_ = x1, len1

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1_, len1_, x2, len2, y)

            # forward / loss
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # export evaluation details
            if params.eval_verbose:
                for i in range(len(len1)):
                    src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                    tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                    s = (
                        f"Equation {n_total.sum().item() + i} "
                        f"({'Valid' if valid[i] else 'Invalid'})\n"
                        f"src={src}\ntgt={tgt}\n"
                    )
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            "equations were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_acc"] = 100.0 * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            scores[f"{data_type}_{task}_acc_{i}"] = (
                100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
            )

    def enc_dec_step_beam_fast(self, data_type, task, scores, size=None):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in [
            "ode_convergence_speed",
            "ode_control",
            "fourier_cond_init",
        ]

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # iterator
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
        eval_size = len(iterator.dataset)

        # save beam results
        beam_log = {}
        hyps_to_eval = []

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # update logs
            for i in range(len(len1)):
                beam_log[i + n_total.sum().item()] = {
                    "src": x1[1 : len1[i] - 1, i].tolist(),
                    "tgt": x2[1 : len2[i] - 1, i].tolist(),
                    "nb_ops": nb_ops[i].item(),
                    "hyps": [],
                }

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # optionally truncate input
            x1_, len1_ = x1, len1

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1_, len1_, x2, len2, y)
            bs = len(len1)

            # forward
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # update stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # update equations that were solved greedily
            for i in range(len(len1)):
                if valid[i]:
                    beam_log[i + n_total.sum().item() - bs]["hyps"].append(
                        (None, None, True)
                    )

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                "Generating solutions ..."
            )

            # generate with beam search
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1_,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=params.max_len,
            )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(
                    sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
                ):
                    hyps_to_eval.append(
                        {
                            "i": i + n_total.sum().item() - bs,
                            "j": j,
                            "score": score,
                            "src": x1[1 : len1[i] - 1, i].tolist(),
                            "tgt": x2[1 : len2[i] - 1, i].tolist(),
                            "hyp": hyp[1:].tolist(),
                            "task": task,
                        }
                    )

        # if the Jacobian is also predicted, only look at the eigenvalue
        if task == "ode_convergence_speed":
            sep_id = env.word2id[env.mtrx_separator]
            for x in hyps_to_eval:
                x["tgt"] = (
                    x["tgt"][x["tgt"].index(sep_id) + 1 :]
                    if sep_id in x["tgt"]
                    else x["tgt"]
                )
                x["hyp"] = (
                    x["hyp"][x["hyp"].index(sep_id) + 1 :]
                    if sep_id in x["hyp"]
                    else x["hyp"]
                )

        # solutions that perfectly match the reference with greedy decoding
        assert all(
            len(v["hyps"]) == 0
            or len(v["hyps"]) == 1
            and v["hyps"][0] == (None, None, True)
            for v in beam_log.values()
        )
        init_valid = sum(
            int(len(v["hyps"]) == 1 and v["hyps"][0][2] is True)
            for v in beam_log.values()
        )
        logger.info(
            f"Found {init_valid} solutions with greedy decoding "
            "(perfect reference match)."
        )

        # check hypotheses with multiprocessing
        eval_hyps = []
        start = time.time()
        logger.info(
            f"Checking {len(hyps_to_eval)} hypotheses for "
            f"{len(set(h['i'] for h in hyps_to_eval))} equations ..."
        )
        with ProcessPoolExecutor(max_workers=20) as executor:
            for output in executor.map(check_hypothesis, hyps_to_eval, chunksize=1):
                eval_hyps.append(output)
        logger.info(f"Evaluation done in {time.time() - start:.2f} seconds.")

        # update beam logs
        for hyp in eval_hyps:
            beam_log[hyp["i"]]["hyps"].append(
                (hyp["hyp"], hyp["score"], hyp["is_valid"])
            )

        # print beam results
        beam_valid = sum(
            int(any(h[2] for h in v["hyps"]) and v["hyps"][0][1] is not None)
            for v in beam_log.values()
        )
        all_valid = sum(int(any(h[2] for h in v["hyps"])) for v in beam_log.values())
        assert init_valid + beam_valid == all_valid
        assert len(beam_log) == n_total.sum().item()
        logger.info(
            f"Found {all_valid} valid solutions ({init_valid} with greedy decoding "
            f"(perfect reference match), {beam_valid} with beam search)."
        )

        # update valid equation statistics
        n_valid = torch.zeros(1000, dtype=torch.long)
        for i, v in beam_log.items():
            if any(h[2] for h in v["hyps"]):
                n_valid[v["nb_ops"]] += 1
        assert n_valid.sum().item() == all_valid

        # export evaluation details
        if params.eval_verbose:

            eval_path = os.path.join(
                params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}"
            )

            with open(eval_path, "w") as f:

                # for each equation
                for i, res in sorted(beam_log.items()):
                    n_eq_valid = sum([int(v) for _, _, v in res["hyps"]])
                    src = idx_to_infix(env, res["src"], input=True).replace("|", " | ")
                    tgt = " ".join(env.id2word[wid] for wid in res["tgt"])
                    s = (
                        f"Equation {i} ({n_eq_valid}/{len(res['hyps'])})\n"
                        f"src={src}\ntgt={tgt}\n"
                    )
                    for hyp, score, valid in res["hyps"]:
                        if score is None:
                            assert hyp is None
                            s += f"{int(valid)} GREEDY\n"
                        else:
                            try:
                                hyp = " ".join(hyp)
                            except Exception:
                                hyp = f"INVALID OUTPUT {hyp}"
                            s += f"{int(valid)} {score :.3e} {hyp}\n"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f.write(s + "\n")
                    f.flush()

            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations "
            "were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size
        scores[f'{data_type}_{task}_xe_loss'] = xe_loss / _n_total 
        scores[f"{data_type}_{task}_beam_acc"] = 100.0 * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            logger.info(
                f"{i}: {n_valid[i].sum().item()} / {n_total[i].item()} "
                f"({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)"
            )
            scores[f"{data_type}_{task}_beam_acc_{i}"] = (
                100.0 * n_valid[i].sum().item() / max(n_total[i].item(), 1)
            )


def convert_to_text(batch, lengths, id2word, params):
    """
    Convert a batch of sequences to a list of text sequences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sequences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(id2word[batch[k, j]])
        sequences.append(" ".join(words))
    return sequences

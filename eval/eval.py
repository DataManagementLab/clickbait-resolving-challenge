import abc
import json
from tqdm.autonotebook import tqdm

class Metric(abc.ABC):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.batch_mode = False

    def run(self, answer, gold_answer):
        pass

    def run_all(self, answers, gold_answers):
        pass

    def acc(self, results):
        return round(sum(results)/len(results), 4)


class ExactMatchMetric(Metric):
    def run(self, answer, gold_answer):
        return 1 if answer == gold_answer else 0


class RougeMetric(Metric):
    def __init__(self, evaluator):
        super().__init__(evaluator)
        from rouge import Rouge
        self.rouge = Rouge()

    def run(self, answer, gold_answer):
        if answer != "" and answer != '-':
            return self.rouge.get_scores(answer, gold_answer)[0]
        return {'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0}, 'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    def acc(self, results):
        rouge_2_f1 = round(sum(r['rouge-2']['f'] for r in results) / len(results), 4)
        rouge_l_f1 = round(sum(r['rouge-l']['f'] for r in results) / len(results), 4)
        return {"Rouge-2 F1": rouge_2_f1, "Rouge-L": rouge_l_f1}


class RougeBatchMetric(RougeMetric):
    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.batch_mode = True

    def run_all(self, answers, gold_answers):
        return self.rouge.get_scores(answers, gold_answers)


class MeteorMetric(Metric):
    def __init__(self, evaluator):
        super().__init__(evaluator)

        import nltk
        nltk.download('omw-1.4')

    def run(self, answer, gold_answer):
        from nltk.translate import meteor
        from nltk.tokenize import word_tokenize

        return round(meteor([word_tokenize(answer)], word_tokenize(gold_answer)), 4)


class BleuMetric(Metric):
    def __init__(self, evaluator):
        super().__init__(evaluator)

    def run(self, answer, gold_answer):
        from nltk.translate import bleu
        from nltk.tokenize import word_tokenize

        return bleu([word_tokenize(answer)], word_tokenize(gold_answer), weights=[(.5, .5)])

    
class BERTScoreMetric(Metric):
    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.batch_mode = True
        from bert_score import score
        self.BERTScorer = score

    def run_all(self, answers, gold_answers):
        _, _, f1_scores = self.BERTScorer(answers, gold_answers, lang='en', rescale_with_baseline=True)
        return f1_scores.tolist()


class ClickbaitResolverEvaluator:
    METRICS = [
        ("Exact Match", ExactMatchMetric),
        ("Rouge", RougeBatchMetric),
        ("Meteor", MeteorMetric),
        ("BLEU", BleuMetric),
        ("BERTScore (F1)", BERTScoreMetric),
    ]

    def __init__(self, metrics=None, use_cuda=False):
        super().__init__()

        if metrics is None:
            metrics = self.METRICS

        assert len(metrics) > 0

        self.use_cuda = use_cuda

        self.metric_handlers = []
        for name, handle in tqdm(metrics, desc="Init metrics"):
            m = handle(self)
            self.metric_handlers.append((name, m))

        self.batch_metrics = [m for m in self.metric_handlers if m[1].batch_mode]
        self.individual_metrics = [m for m in self.metric_handlers if not m[1].batch_mode]

    def run(self, answers, gold_answers):
        results = [{} for _ in range(len(answers))]

        assert len(answers) == len(gold_answers)

        # Batch metrics
        for metric_name, m in tqdm(self.batch_metrics, desc="calculate batch"):
            metric_results = m.run_all(answers, gold_answers)
            for result, metric_result in zip(results, metric_results):
                result[metric_name] = metric_result

        # Other metrics
        for result, answer, gold_answer in tqdm(zip(results, answers, gold_answers), desc="calculate individual", total=len(gold_answers)):
            for metric_name, m in self.individual_metrics:
                result[metric_name] = m.run(answer, gold_answer)

        agg_results = {}
        for name, m in tqdm(self.metric_handlers, desc="aggregate"):
            results_by_metric = [r[name] for r in results]
            agg_results[name] = m.acc(results_by_metric)

        return agg_results, results

    def run_file(self, answer_filename, gold_filename):
        with open(answer_filename, "r") as answer_file:
            answers_from_file = {a['id']:a['answer'] for a in json.load(answer_file)}

        answers = []
        gold_answers = []
        with open(gold_filename, "r") as gold_file:
            for ga in json.load(gold_file):
                gold_answers.append(ga["answer"])
                answers.append(answers_from_file.get(ga["id"], "-"))

        return self.run(answers, gold_answers)

    def print_results(self, agg_results, results, print_details=False):
        print("")
        print(agg_results)
        if print_details:
            print("\nDetails:")
            print(results)


if __name__ == "__main__":
    evaluator = ClickbaitResolverEvaluator(use_cuda=False)
    agg_results, results = evaluator.run(["I'm blue da be dee", "Bass bass", "abc def"], ["da be dee da be dai", "wir brauchen Bass", "abc def"])
    evaluator.print_results(agg_results, results)

    #agg_results, results = evaluator.run_file("../data/baseline_results/first_sentence/train.json", "../data/final_train.json")
    #evaluator.print_results(agg_results, results, False)

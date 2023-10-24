class Metrics():
    def __init__(self, category_labels, indicators_labels):
        self.category_labels = category_labels
        self.indicators_labels = indicators_labels
        self.metrics_framework = {}
        self.metrics = {}
        self.global_metrics = {}
        for label in category_labels:
            entry = {}
            for indicator in indicators_labels:
                entry[indicator] = 0
            self.metrics[label] = entry
            self.global_metrics[label] = entry

    def accumulateMetrics(self):
        for label in self.category_labels:
            for indicator in self.indicators_labels:
                self.global_metrics[label][indicator] = self.global_metrics[label][indicator] + self.metrics[label][indicator]
    
    def resetMetrics(self):
        self.metrics = {}
        for label in self.category_labels:
            entry = {}
            for indicator in self.indicators_labels:
                entry[indicator] = 0
            self.metrics[label] = entry

    def addMetrics2File(self,file):
        self.metrics_framework[file] = self.metrics

    def addValue2Indicator(self, category, indicator):
        entry = self.metrics[category]
        entry[indicator] = entry[indicator] + 1
        self.metrics[category] = entry

    def getMetricsFramework(self):
        return self.metrics_framework
    
    def calculateMetrics(self):
        kk = self.metrics.keys
        vv = self.metrics.values
        for category in self.metrics:
            entry = self.metrics[category]
            if entry["true_positives"]!= 0 and entry["false_positives"]!=0:
                entry["precision"] = entry["true_positives"]/(entry["true_positives"]+entry["false_positives"])
            else:
                entry["precision"] = 0
            if (entry["true_positives"]!=0 and entry["false_negatives"]!=0):
                entry["recall"] = entry["true_positives"]/(entry["true_positives"]+entry["false_negatives"])
            else: 
                entry["recall"] = 0
            if (entry["precision"] != 0 and entry["recall"]!=0):
                entry["f-score"] = (2 * entry["precision"] * entry["recall"]) / (entry["precision"]+entry["recall"]) 
            else:
                entry["f-score"] = 0
            self.metrics[category] = entry
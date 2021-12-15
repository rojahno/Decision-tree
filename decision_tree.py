import numpy
import numpy as np
import pandas as pd
import copy

data_set_1 = "data_1.csv"
data_set_2 = "data_2.csv"


class DecisionTree:

    def __init__(self):
        self.entropy_set = None
        self.decider = []
        self.rules = []
        self.tree = None
        self.i = 0

    def fit(self, x, y):
        """
        Generates a decision tree for classification
        
        Args:
            x (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y:
        """
        self.tree = self.build_tree(x, y, tree=None)

    def build_tree(self, x, y, tree=None):
        """
        Builds the decition tree
        Args:
            x: The attributes of the dataset
            y: The outcome of the different attributes
            tree:

        Returns: A dictionary with the max gain attribute and its value

        """
        # Joins x and y to one dataframe
        data_frame = x
        key = y.name
        data_frame[y.name] = y
        nr_colum = len(x.columns)

        # Checks if there is few columns left and guesses
        if nr_colum == 2 and len(y.unique()) > 1:
            tree = dict()
            column_value = x.columns
            column_name = column_value[0]
            y_value = y.to_numpy()[0]
            tree[column_name] = y_value
            return tree

        if self.entropy_set is None:
            self.set_total_entropy(data_frame, key)

        # Get entropy and average information gain for available columns
        attribute_dict = dict()
        entropy_dict = dict()
        for name in data_frame.keys()[:-1]:  # Checks all columns except the outcome column
            entropy_dict[name] = self.find_attribute_entropy(data_frame, name)
            attribute_dict[name] = self.find_attribute_average(data_frame, name)

        # Calculates the gain for available columns
        gain_dict = dict()
        for name in attribute_dict:
            attribute = attribute_dict[name]
            gain = self.find_gain(attribute)
            gain_dict[name] = gain

        # Get the value with the highest gain
        max_gain_attribute = max(gain_dict, key=gain_dict.get)

        # If the tree is none, initialize the tree with the max gain attribute as hash
        if tree is None:
            tree = dict()
            tree[max_gain_attribute] = {}

        # Build the tree and calls fit() recursively until it reaches an outcome
        for attribute in attribute_dict[max_gain_attribute]:
            value = abs(entropy_dict[max_gain_attribute][attribute])
            if value == 0:
                attribute_column = data_frame[(data_frame[max_gain_attribute] == attribute)]
                value = attribute_column[key].unique()
                tree[max_gain_attribute][attribute] = value[0]
            else:
                new_data_frame = data_frame[(data_frame[max_gain_attribute] == attribute)]
                x = new_data_frame.drop(columns=[key, max_gain_attribute])
                y = new_data_frame[key]
                tree[max_gain_attribute][attribute] = self.build_tree(x, y)

        return tree

    def predict(self, x: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            x (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        prediction_list = []
        prediction_column = "Prediction"
        for (label, series) in x.iterrows():
            outcome = self.traverse(self.tree, series)
            if outcome is None:
                prediction_list.append("Undefined")
            else:
                prediction_list.append(outcome)

        # Adds the prediction list to the dataframe to get the same indexes for the predictions
        x[prediction_column] = prediction_list
        y_prediction = x[prediction_column]
        return y_prediction

    def traverse(self, tree: dict, row: pd.Series):
        """
        Traverses the tree
        Args:
            tree: The decision tree
            row:

        Returns: The target node we are searching for.

        """
        category = list(tree.keys())[0]
        category_map = tree[category]
        if type(category_map) is str:
            return category_map

        if row.get(category) not in category_map.keys():
            return None

        target_node = category_map[row.get(category)]

        if type(target_node) == dict:
            return self.traverse(target_node, row)
        else:
            return target_node

    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        rules_list = []
        term_list = []
        self.build_rules(self.tree, rules_list, term_list)
        return rules_list

    def build_rules(self, tree: dict, rule_list: list, term_list):
        """
        Builds the rules of the decision tree
        Args:
            tree:
            rule_list:
            term_list:
        """

        if type(tree) == str:
            rule_list.append(tuple([term_list, tree]))
        else:
            for (branch_key, branch_value) in tree.items():
                if type(tree[branch_key]) == str:
                    term_list.append(tuple([branch_key, branch_value]))
                    rule_list.append(tuple([term_list, tree[branch_key]]))
                else:
                    for (next_key, next_value) in tree[branch_key].items():
                        new_term_list = copy.deepcopy(term_list)
                        new_term_list.append(tuple([branch_key, next_key]))
                        self.build_rules(next_value, rule_list, new_term_list)

    def set_total_entropy(self, data_frame, attribute):
        deciders = data_frame[attribute].unique()
        for target in deciders:
            value = len(data_frame[(data_frame[attribute] == target)])
            self.decider.append(value)
        outcome = np.array(self.decider)
        self.entropy_set = entropy(outcome)

    def find_attribute_entropy(self, data_frame, attribute):
        deciders = data_frame.keys()[-1]
        target_variables = data_frame[deciders].unique()  # The "Yes" and "No"
        variables = data_frame[attribute].unique()  # The different variables in the column
        variable_entropy = dict()
        for variable in variables:
            variable_list = []
            for target in target_variables:
                nr = len(data_frame[(data_frame[attribute] == variable) & (data_frame[deciders] == target)])
                variable_list.append(nr)
            count = np.array(variable_list)
            entropy_score = entropy(count)
            variable_entropy[variable] = entropy_score
        return variable_entropy

    def find_attribute_average(self, data_frame, attribute):
        deciders = data_frame.keys()[-1]
        target_variables = data_frame[deciders].unique()  # The "Yes" and "No"
        variables = data_frame[attribute].unique()  # The different variables in the column
        variable_entropy = dict()
        for variable in variables:
            variable_list = []
            for target in target_variables:
                nr = len(data_frame[(data_frame[attribute] == variable) & (data_frame[deciders] == target)])
                variable_list.append(nr)
            count = np.array(variable_list)
            entropy_score = entropy(count)
            average_information = self.find_average_information_entropy(count, entropy_score)
            variable_entropy[variable] = average_information
        return variable_entropy

    def find_average_information_entropy(self, count, entropy):
        total_attribute = sum(count)
        total_set = sum(self.decider)
        average = ((total_attribute / total_set) * entropy)
        return average

    def find_gain(self, attributes):
        """
        Find the total gain of each attribute
        Args:
            attributes: The attributes we want to find the gain for.

        Returns:The gain

        """
        average = 0
        for name in attributes:
            average = average + (attributes[name])
        return self.entropy_set - average

    def print_rules(self, outcome: str):
        tree_rules = self.get_rules()
        for rules, label in tree_rules:
            conjunction = ' ∩ '.join(f'{attr}={value}' for attr, value in rules)
            print(f'{"✅" if label == outcome else "❌"} {conjunction} => {label}')

    # --- Some utility functions


def accuracy(y_true: pd.Series, y_pred: pd.Series):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
        """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """

    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


def first_example():
    # Retrieve the data set and get a DataFrame
    data = pd.read_csv(data_set_1)
    print(f'Full data set: \n{data} \n')

    # Remove the play tennis column and assign the remaining values to x
    x = data.drop(columns=['Play Tennis'])

    # Assign the outcome column to y
    y = data['Play Tennis']
    model_1 = DecisionTree()
    model_1.fit(x, y)
    rule = model_1.get_rules()
    print(f'Rules: {rule}')
    x = data.drop(columns=['Play Tennis'])

    # Verify that it perfectly fits the training set
    print(f'Accuracy: {accuracy(y_true=y, y_pred=model_1.predict(x)) * 100 :.1f}%\n')
    model_1.print_rules("Yes")


def second_example():
    data_2 = pd.read_csv(data_set_2)
    data_2 = data_2.drop(columns=['Founder Zodiac'])  # Drops data which creates noise for the training.
    # print(f'Data:\n{data_2}')
    data_2_train = data_2.query('Split == "train"')
    data_2_valid = data_2.query('Split == "valid"')
    data_2_test = data_2.query('Split == "test"')

    X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
    X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
    X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome
    data_2.Split.value_counts()

    # Fit model
    model_2 = DecisionTree()
    model_2.fit(X_train, y_train)
    print(f'Train: {accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
    print(f'Valid: {accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
    print(f'Test: {accuracy(y_test, model_2.predict(X_test)) * 100 :.1f}%')
    model_2.print_rules("success")


def main():
    # first_example()
    second_example()


if __name__ == "__main__":
    main()

import csv

class Transaction:
    def __init__(self, items):
        self.items = set(items)


# for get all item function, there is a one line lambda expression
flatten = lambda l: [item for sublist in l for item in sublist]
class Apriori:
    def __init__(self, list_transactions):
        self.transactions = list_transactions
        self.all_items = set()
        for trn in self.transactions:
            self.all_items |= trn.items
        pass

    def get_support(self, set_items):
        # !!! Write code to return support of set_items
        
        pass
   
    def get_combinations(self, items, num):
        if num:
            for i in range(num -1, len(items)):
                for cn in get_combinations(items[:i], num -1):
                    yield cn +(items[i], )
        else:
            yield tuple()
        pass

    def get_all_items(self, itemsets):
        # !!! Write code to merge all the itemsets and return the current items
        # all_items =[item for sublist in test_data for item in sublist]
        return flatten(itemsets)
        pass

    # using magical bitmap
    def is_bit_set(num, bit):
        return num&(1 <<bit) > 0

    def get_subsets(self, items):
        # !!! Write code to return all subsets of "items"
        sets =[]
        for i in range(1 <<len(items)):
            subset =[items[bit] for bit in range(len(items)) if is_bit_set(i, bit)]
            sets.append(subset)
        return sets
        pass

    def get_rules(self, min_sup, min_con):
        var_a = self.get_combinations(self.all_items, 1)
        num = 2
        freq_items = []
        supports = []
        while True:
            var_b = map(lambda x: self.get_support(x), var_a)
            chosen_itemset = []
            chosen_supports = []
            for index, items in enumerate(var_a):
                if var_b[index] >= min_sup:
                    chosen_itemset.append(items)
                    chosen_supports.append(var_b[index])
            freq_items.extend(chosen_itemset)
            supports.extend(chosen_supports)
            var_c = self.get_all_items(chosen_itemset)
            if num > len(var_c):
                break
            var_a = self.get_combinations(var_c, num)
            num += 1
        print "Frequent Itemsets"
        for item, support in sorted(zip(freq_items, supports), key=lambda x: x[1]):
            print ",".join(item), support

        rules = []
        confidences = []
        print "Rules"
        for items in freq_items:
            subsets = self.get_subsets(items)
            support = self.get_support(items)
            for var_d in subsets:
                if len(items - var_d) == 0:
                    continue
                support_A = self.get_support(var_d)
                if support / support_A > min_con:
                    rules.append((var_d, items - var_d))
                    confidences.append(support / support_A)

        for rule, confidence in sorted(zip(rules, confidences), key=lambda x: x[1]):
            print ",".join(rule[0]), "=>", ",".join(rule[1]), confidence

if __name__ == "__main__":
    test_data = [
        ['bread', 'milk'],
        ['bread', 'diaper', 'beer', 'egg'],
        ['milk', 'diaper', 'beer', 'cola'],
        ['bread', 'milk', 'diaper', 'beer'],
        ['bread', 'milk', 'diaper', 'cola'],
    ]

    transactions = []
    test = False

    if test:
        for row in test_data:
            transactions.append(Transaction(set(row)))
    else:
        with open("apriori_data.csv", "r") as f:
            for row in csv.reader(f):
                transactions.append(Transaction(set(row)))

    a = Apriori(transactions)
    a.get_rules(min_sup=0.15, min_con= 0.6)




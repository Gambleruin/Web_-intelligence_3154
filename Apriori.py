import csv
import pdb
class Transaction:
    def __init__(self, items):
        self.items = set(items)

class Apriori:
    def __init__(self, list_transactions):
        self.flatten = lambda l: [item for sublist in l for item in sublist]
        self.transactions = list_transactions
        self.all_items = set()
        for trn in self.transactions:
            self.all_items |=trn.items
        pass

    def get_support(self, item):
        # !!! Write code to return support of set_items
        count =0
        N= len(self.all_items)
        item_ ={elem for elem in item}

        for i in range(len(self.transactions)):
            if item_ < self.transactions[i].items:
                count =count +1
        return count/N
        
        pass

    def get_combinations(self, items, num):
        # !!! Write code to return "num" number of combinations of items
        item_list =list(items)
        if num:
            for i in range(num -1, len(items)):
                for cn in self.get_combinations(item_list[:i], num -1):
                    yield cn +(item_list[i],)
        else:
            yield tuple()  
        pass

    def get_all_items(self, itemsets):
        fat_list =self.flatten(itemsets)
        all_items =set(fat_list)
        return all_items
        pass

    def is_bit_set(self, num, bit):
        return num&(1 <<bit) >0

    def get_subsets(self, items):
        # !!! Write code to return all subsets of "items"
        sets =[]
        for i in range(1 <<len(items)):
            subset =[items[bit] for bit in range(len(items)) if self.is_bit_set(i, bit)]
            sets.append(subset)

        # the empty list better to be removed
        subsets =[x for x in sets if x !=[]]
        return subsets
        pass

    def get_rules(self, min_sup, min_con):
        var_a = list(self.get_combinations(self.all_items, 1))
        num = 2
        freq_items = []
        supports = []

        while True:
            var_b = list(map(lambda x: self.get_support(x), var_a))
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
            var_a = list(self.get_combinations(var_c, num))
            num += 1
            # print(num, len(var_c))
            
         
        print('the loop is over')   
        # print ("Frequent Itemsets")
        '''
        for item, support in sorted(zip(freq_items, supports), key=lambda x: x[1]):
            print (",".join(item), support)
        '''

        rules = []
        confidences = []
        # print ("Rules")
        # print(var_a, '\n')
        # print(freq_items, '\n','\n\n')


        '''
        for item in freq_items:
            print(item)
            subsets = self.get_subsets(item)
            print('\n', subsets)

        '''     
        for items in freq_items:
            items_ =set(items)
            subsets = self.get_subsets(items)
            # print(subsets_)
            # print(subsets, '\n')
            support = self.get_support(items)
            for var_d in subsets:
                var_d_ =set(var_d)
                # var_d_ =self.get_all_items(var_d)
                # print(var_d_, items_)
                if len(items_ - var_d_) == 0:
                    continue
                support_A = self.get_support(var_d_)

                if support / support_A > min_con:
                    rules.append((var_d_, items_ - var_d_))
                    confidences.append(support / support_A)

        for rule, confidence in sorted(zip(rules, confidences), key=lambda x: x[1]):
            print (",".join(rule[0]), "=>", ",".join(rule[1]), confidence)

        


if __name__ == "__main__":
    test_data = [
        ['bread', 'milk'],
        ['bread', 'diaper', 'beer', 'egg'],
        ['milk', 'diaper', 'beer', 'cola'],
        ['bread', 'milk', 'diaper', 'beer'],
        ['bread', 'milk', 'diaper', 'cola'],
    ]

    transactions = []
    test = True

    if test:
        for row in test_data:
            transactions.append(Transaction(row))
    else:
        with open("apriori_data.csv", "r") as f:
            for row in csv.reader(f):
                transactions.append(Transaction(row))

    a = Apriori(transactions)
    a.get_rules(min_sup=0.15, min_con= 0.6)


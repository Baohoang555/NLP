import re
from collections import defaultdict, Counter
import numpy as np

class PoetryCorpus:
    def __init__(self, poems_text):
        """Initializes the PoetryCorpus with the given poems text."""
        self.poems = poems_text
        self.vocab_single = set()  # từ đơn
        self.vocab_bigram = set()  # từ ghép 2 (SỬA: vocal → vocab)
        self.bigram_freq = Counter()
        self.trigram_freq = Counter()  # SỬA: trigarm → trigram
        self.word_freq = Counter()
        
    def tokenize(self, text):
        text = re.sub(r'[,.:;!?]', '', text.lower())  # SỬA: sup → sub
        return text.split()
    
    def build_corpus(self, dictionary_words=None):
        all_words = []
        for poem in self.poems:
            words = self.tokenize(poem)
            all_words.extend(words)
            
            # từ đơn
            for word in words:
                if dictionary_words is None or word in dictionary_words:
                    self.vocab_single.add(word)
                    self.word_freq[word] += 1
                    
            # từ ghép 2
            for i in range(len(words) - 1):
                bigram = f"{words[i]}_{words[i+1]}"
                if dictionary_words is None or (words[i] in dictionary_words and words[i+1] in dictionary_words):
                    self.vocab_bigram.add(bigram)  # SỬA: vocal → vocab
                    self.bigram_freq[(words[i], words[i+1])] += 1
                    
            # từ ghép 3
            for i in range(len(words) - 2):
                self.trigram_freq[(words[i], words[i+1], words[i+2])] += 1

        print(f"Corpus built: {len(self.vocab_single)} từ đơn, {len(self.vocab_bigram)} từ ghép 2, {len(self.trigram_freq)} từ ghép 3.")
        return self  # Thêm return để chain method


class MinEditDistance:
    def __init__(self, ins_cost=1, del_cost=1, sub_cost=2):
        self.ins_cost = ins_cost
        self.del_cost = del_cost
        self.sub_cost = sub_cost
    
    def compute(self, source, target, return_backtrace=False):
        n = len(source)
        m = len(target)
        D = np.zeros((n+1, m+1))
        backtrace = [[None for _ in range(m+1)] for _ in range(n+1)]
        
        for i in range(1, n+1):
            D[i][0] = D[i-1][0] + self.del_cost
            backtrace[i][0] = 'DEL'
            
        for j in range(1, m+1):  # SỬA: i → j
            D[0][j] = D[0][j-1] + self.ins_cost
            backtrace[0][j] = 'INS'
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                if source[i-1] == target[j-1]:  # SỬA: target == target → chỉ 1 lần
                    D[i][j] = D[i-1][j-1]
                    backtrace[i][j] = 'MATCH'
                else:
                    del_cost = D[i-1][j] + self.del_cost
                    ins_cost = D[i][j-1] + self.ins_cost
                    sub_cost = D[i-1][j-1] + self.sub_cost
                    
                    min_cost = min(del_cost, ins_cost, sub_cost)
                    D[i][j] = min_cost
                    
                    if min_cost == sub_cost:
                        backtrace[i][j] = 'SUB'
                    elif min_cost == del_cost:
                        backtrace[i][j] = 'DEL'
                    else:
                        backtrace[i][j] = 'INS'
                        
        if return_backtrace:
            return D[n][m], self._get_alignment(source, target, backtrace)  # SỬA: thứ tự params
        return D[n][m]
    
    def _get_alignment(self, source, target, backtrace):
        alignment = []
        i = len(source)
        j = len(target)
        
        while i > 0 or j > 0:
            if i == 0:
                alignment.append(('', target[j-1], 'INS'))
                j -= 1  # SỬA: THIẾU dòng này!
            elif j == 0:
                alignment.append((source[i-1], '', 'DEL'))
                i -= 1
            else:
                op = backtrace[i][j]
                if op == 'MATCH':
                    alignment.append((source[i-1], target[j-1], 'MATCH'))
                    i -= 1
                    j -= 1
                elif op == 'SUB':
                    alignment.append((source[i-1], target[j-1], 'SUB'))
                    i -= 1
                    j -= 1
                elif op == 'DEL':
                    alignment.append((source[i-1], '', 'DEL'))
                    i -= 1
                else:
                    alignment.append(('', target[j-1], 'INS'))
                    j -= 1
        return list(reversed(alignment))


class NGramModel:
    def __init__(self, corpus):
        self.corpus = corpus 
        
    def bigram_prob(self, w1, w2):
        bigram_count = self.corpus.bigram_freq[(w1, w2)]  # SỬA: bigaram → bigram
        w1_count = self.corpus.word_freq[w1]
        if w1_count == 0:
            return 0
        return bigram_count / w1_count
    
    def trigram_prob(self, w1, w2, w3):
        trigram_count = self.corpus.trigram_freq[(w1, w2, w3)]
        bigram_count = self.corpus.bigram_freq[(w1, w2)]
        if bigram_count == 0:
            return 0
        return trigram_count / bigram_count
    
    def sentence_prob_bigram(self, words):
        prob = 1.0 
        for i in range(len(words) - 1):
            p = self.bigram_prob(words[i], words[i+1])
            if p == 0:
                p = 1e-10  # SỬA: 1e - 10 → 1e-10
            prob *= p
        return prob
    
    def sentence_prob_trigram(self, words):
        prob = 1.0 
        for i in range(len(words) - 2):
            p = self.trigram_prob(words[i], words[i+1], words[i+2])
            if p == 0:
                p = 1e-10  # SỬA: 1e - 10 → 1e-10
            prob *= p
        return prob


# SỬA: DI CHUYỂN RA NGOÀI CLASS! (Đây là lỗi chính)
def noisy_channel_search(query, corpus, med_model, top_k=5):  # SỬA: querry → query
    """
    Tìm kiếm với Noisy Channel Model
    P(intended|typed) ∝ P(typed|intended) * P(intended)
    """
    query_words = corpus.tokenize(query)  # SỬA: querry → query
    results = []
    
    for poem in corpus.poems:
        for line in poem.split('\n'):
            if not line.strip():
                continue
            
            line_words = corpus.tokenize(line)
            if len(line_words) == 0:
                continue
            
            # Tính edit distance
            ed = med_model.compute(' '.join(query_words), ' '.join(line_words))
            
            # P(typed|intended) ~ 1/ed
            likelihood = 1 / (1 + ed)
            
            # P(intended) ~ frequency
            prior = sum([corpus.word_freq.get(w, 1) for w in line_words]) / len(line_words)
            
            score = likelihood * prior
            results.append((line.strip(), ed, score))
    
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


# ===== MAIN =====
if __name__ == "__main__":
    poems = [
        "Khi bão độ bổ xông gió nổi lên\nSóng cuồn cuộn dâng trào như núi cao",
        "Gió đông về mùa xuân đã tới\nHoa nở rộ khắp nơi đồng nội",
        "Trăng sáng chiếu rọi bóng người\nGió nhẹ thổi qua cành liễu"
    ]
    

    
    # Bước 1: Tạo corpus
    print("\n1. TẠO CORPUS")
    print("-" * 60)
    corpus = PoetryCorpus(poems)
    corpus.build_corpus()
    
    # Bước 2: Test với câu đầu vào
    print("\n2. TEST VỚI CÂU: 'Khi bão dộ bộ, xống gió nổi lên'")
    print("-" * 60)
    query = "Khi bão dộ bộ, xống gió nổi lên"
    
    # 2.1 Noisy Channel (MED)
    print("\n2.1. NOISY CHANNEL (MED Edit Distance):")
    med = MinEditDistance()
    results = noisy_channel_search(query, corpus, med, top_k=3)
    
    for i, (line, ed, score) in enumerate(results, 1):
        print(f"\nTop {i}: {line}")
        print(f"  Edit Distance: {ed}")
        print(f"  Score: {score:.4f}")
    
    # 2.2 N-Gram
    print("\n2.2. N-GRAM MODEL:")
    ngram = NGramModel(corpus)
    query_words = corpus.tokenize(query)
    
    if len(query_words) >= 2:
        bigram_prob = ngram.sentence_prob_bigram(query_words)
        print(f"  Bigram probability: {bigram_prob:.6e}")
    
    if len(query_words) >= 3:
        trigram_prob = ngram.sentence_prob_trigram(query_words)
        print(f"  Trigram probability: {trigram_prob:.6e}")
    
    # Bước 3: Cải tiến với backtrace
    print("\n3. CẢI TIẾN VỚI BACKTRACE")
    print("-" * 60)
    target = "Khi bão độ bổ xông gió nổi lên"
    ed, alignment = med.compute(query, target, return_backtrace=True)
    
    print(f"Edit Distance: {ed}")
    print("Alignment:")
    for src, tgt, op in alignment:
        if op == 'MATCH':
            print(f"  ✓ '{src}' == '{tgt}' (MATCH)")
        elif op == 'SUB':
            print(f"  ✗ '{src}' → '{tgt}' (SUBSTITUTE)")
        elif op == 'DEL':
            print(f"  ✗ '{src}' → '' (DELETE)")
        elif op == 'INS':
            print(f"  ✗ '' → '{tgt}' (INSERT)")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
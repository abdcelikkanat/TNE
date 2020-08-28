from gensim.models.word2vec import *
from ext.gensim_wrapper.models.keyedvectors import *

try:
    import pyximport

    from numpy import get_include
    models_dir = os.path.dirname(__file__) or os.getcwd()

    pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
    #import pyximport; pyximport.install()
    from ext.gensim_wrapper.models.word2vec_inner import train_batch_sg_community
except ImportError:
    raise ValueError("An error occurred while loading the optimized version!")


class Word2VecWrapper(Word2Vec):

    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False):

        super(Word2VecWrapper, self).__init__(sentences, size, alpha, window, min_count,
                                              max_vocab_size, sample, seed, workers, min_alpha,
                                              sg, hs, negative, cbow_mean, hashfxn, iter, null_word,
                                              trim_rule, sorted_vocab, batch_words, compute_loss)

    def train_community(self, number_of_communities, sentences, comm_embedding_size, total_examples=None,
                        total_words=None, epochs=None, start_alpha=None, end_alpha=None, word_count=0,
                        queue_factor=2, report_delay=1.0, compute_loss=None):

        total_examples = self.corpus_count
        epochs = self.iter
        #self.reset_community_weights(number_of_communities, comm_embedding_size)
        self.negative = 10

        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        if compute_loss:
            self.compute_loss = compute_loss
        self.running_training_loss = 0

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative, self.window)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.syn0_community):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of sentences in the training corpus is missing. Did you load the model via KeyedVectors.load_word2vec_format?"
                "Models loaded via load_word2vec_format don't support further training. "
                "Instead start with a blank model, scan_vocab on the new corpus, intersect_word2vec_format with the old model, then train.")

        if total_words is None and total_examples is None:
            raise ValueError("You must specify either total_examples or total_words, for proper alpha and progress calculations. The usual value is total_examples=model.corpus_count.")
        if epochs is None:
            raise ValueError("You must specify an explict epochs count. The usual value is epochs=model.iter.")
        start_alpha = start_alpha or self.alpha
        end_alpha = end_alpha or self.min_alpha

        job_tally = 0

        if epochs > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, epochs)
            total_words = total_words and total_words * epochs
            total_examples = total_examples and total_examples * epochs

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_community_job(sentences, alpha, (work, neu1) ,number_of_communities)
                #tally, raw_tally = self._do_train_job(sentences, alpha, (work, neu1))
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = start_alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warning(
                    "Effective 'alpha' higher than previous training cycles"
                )
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if end_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
                        next_alpha = max(end_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warning(
                "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
            )

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning(
                "supplied example count (%i) did not equal expected count (%i)", example_count, total_examples
            )
        if total_words and total_words != raw_word_count:
            logger.warning(
                "supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words
            )

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def _do_train_community_job(self, sentences, alpha, inits, number_of_communities):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg_community(self, sentences, alpha, work, self.compute_loss, number_of_communities)
        else:
            raise ValueError("It has not been implemented for CBOW!")
            #tally += train_batch_cbow_topic(self, sentences, alpha, work, neu1, self.compute_loss)
        return tally, self._raw_word_count(sentences)

    def reset_community_weights(self, number_of_communities, comm_embedding_size):
        self.layer1_size = comm_embedding_size
        self.vector_size = comm_embedding_size
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
        self.wv.syn0_community = empty((number_of_communities, self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        #for i in xrange(len(self.wv.vocab)):
        for i in xrange(number_of_communities):
            # construct deterministic seed from word AND seed argument
            self.wv.syn0_community[i] = self.seeded_vector(str(i) + str(self.seed))
            self.wv.syn0[i] = self.seeded_vector(str(i) + str(self.seed))
        if self.hs:
            self.syn1 = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
            self.syn1neg_community = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        self.wv.syn0norm_community = None
        self.syn0_lockf_community = ones(number_of_communities, dtype=REAL)  # zeros suppress learning
        self.wv.syn0norm = None
        self.syn0_lockf = ones(number_of_communities, dtype=REAL)  # zeros suppress learning

    def initialize_word_vectors(self):
        self.wv = KeyedVectorsWrapper()

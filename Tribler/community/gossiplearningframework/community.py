from Tribler.dispersy.authentication import MemberAuthentication
from Tribler.dispersy.community import Community
from Tribler.dispersy.conversion import DefaultConversion
from Tribler.dispersy.message import DropMessage
from Tribler.dispersy.distribution import DirectDistribution
from Tribler.dispersy.member import Member
from Tribler.dispersy.resolution import PublicResolution

from Tribler.dispersy.dispersy import Dispersy
from Tribler.dispersy.dispersydatabase import DispersyDatabase
from Tribler.dispersy.distribution import FullSyncDistribution, LastSyncDistribution
from Tribler.dispersy.message import Message, DelayMessageByProof
from Tribler.dispersy.resolution import LinearResolution
from Tribler.dispersy.destination import CommunityDestination

from conversion import JSONConversion
import numpy as np
from collections import deque

from payload import MessagePayload, GossipMessage
from models.logisticregression import LogisticRegressionModel
from models.adalineperceptron import AdalinePerceptronModel
from models.p2pegasos import P2PegasosModel
from Tribler.community.gossiplearningframework.youtube_classifier.features import create_features,\
    load_words
from Tribler.community.gossiplearningframework.youtube_classifier.dict_vectorizer import DictVectorizer

# Send messages every 1 seconds.
DELAY=1.0

# Start after 15 seconds.
INITIALDELAY=15.0

# Model queue size.
MODEL_QUEUE_SIZE=10

if __debug__:
    from Tribler.dispersy.dprint import dprint

class GossipLearningCommunity(Community):
    @classmethod
    def get_master_members(cls):
        master_key = "3081a7301006072a8648ce3d020106052b810400270381920004039a2b5690996f961998e72174a9cf3c28032de6e50c810cde0c87cdc49f1079130f7bcb756ee83ebf31d6d118877c2e0080c0eccfc7ea572225460834298e68d2d7a09824f2f0150718972591d6a6fcda45e9ac854d35af1e882891d690b2b2aa335203a69f09d5ee6884e0e85a1f0f0ae1e08f0cf7fbffd07394a0dac7b51e097cfebf9a463f64eeadbaa0c26c0660".decode("HEX")
        master = Member(master_key)
        return [master]

    @classmethod
    def load_community(cls, master, my_member):
        dispersy_database = DispersyDatabase.get_instance()
        try:
            dispersy_database.execute(u"SELECT 1 FROM community WHERE master = ?", (master.database_id,)).next()
        except StopIteration:
            return cls.join_community(master, my_member, my_member)
        else:
            return super(GossipLearningCommunity, cls).load_community(master)

    def __init__(self, master):
        super(GossipLearningCommunity, self).__init__(master)
        if __debug__: dprint('gossiplearningcommunity ' + self._cid.encode("HEX"))
        
        load_words()

        # Periodically we will send our data to other node(s).
        self._dispersy.callback.register(self.active_thread, delay=INITIALDELAY)

        # Stats
        self._msg_count = 0

        # These should be loaded from a database, x and y are stored only locally
        self._x = None
        self._y = None

        self._model_queue = deque(maxlen=MODEL_QUEUE_SIZE)

        # Initial model
        # initmodel = AdalinePerceptronModel()
        # initmodel = LogisticRegressionModel()
        initmodel = P2PegasosModel()

        self._model_queue.append(initmodel)

    def initiate_meta_messages(self):
        """Define the messages we will be using."""
        return [Message(self, u"modeldata",
                MemberAuthentication(encoding="sha1"), # Only signed with the owner's SHA1 digest
                PublicResolution(),
                DirectDistribution(),
#                FullSyncDistribution(), # Full gossip
                CommunityDestination(node_count=1), # Reach only one node each time.
                MessagePayload(),
                self.check_model,
                self.on_receive_model)]

    def initiate_conversions(self):
        return [DefaultConversion(self),
                JSONConversion(self)]

    def send_messages(self, messages):
        meta = self.get_meta_message(u"modeldata")

        send_messages = []

        for message in messages:
            assert isinstance(message, GossipMessage)

            # Create and implement message with 3 parameters
            send_messages.append(meta.impl(authentication=(self._my_member,),
                                           distribution=(self.global_time,),
                                           payload=(message,)))
        # if __debug__: dprint("GOSSIP: calling self._dispersy.store_update_forward(%s, store = False, update = False, forward = True)." % send_messages)
        self._dispersy.store_update_forward(send_messages, store = False, update = False, forward = True)

    def active_thread(self):
        """
        Active thread, send a message and wait delta time.
        """
        while True:
            # Send the last model in the queue.
            self.send_messages([self._model_queue[-1]])
            yield DELAY

    def check_model(self, messages):
        """
        One or more models have been received, we check them for validity.
        This is a generator function and we can either forward a message or drop it.
        """
        for message in messages:
            if isinstance(message.payload.message, GossipMessage):
                age = message.payload.message.age
                if not type(age) == int or age < 0:
                    yield DropMessage(message, "Age must be a nonnegative integer in this protocol.")
                else:
                    yield message # Accept message.
            else:
                yield DropMessage(message, "Message must be a Gossip Message.")

    def on_receive_model(self, messages):
        """
        One or more models have been received from other peers so we update and store.
        """
        for message in messages:
            # Stats.
            self._msg_count += 1
            if __debug__: dprint(("Received message:", message.payload.message))

            msg = message.payload.message

            assert isinstance(msg, GossipMessage)

            # Database not yet loaded.
            if self._x == None or self._y == None:
                if __debug__: dprint("Database not yet loaded.")
                continue

            # Create and store new model using one strategy.
            self._model_queue.append(self.create_model_mu(msg, self._model_queue[-1]))

    def update(self, model):
        """Update a model using all local training examples."""
        for x, y in zip(self._x, self._y):
            model.update(x, y)

    def create_model_rw(self, m1, m2):
        self.update(m1)
        return m1

    def create_model_mu(self, m1, m2):
        m1.merge(m2)
        self.update(m1)
        return m1

    def create_model_um(self, m1, m2):
        self.update(m1)
        self.update(m2)
        m1.merge(m2)
        return m1

    def predict(self, x):
        """Predict with the last model in the queue."""
        return self._model_queue[-1].predict(x)

    def user_input(self, is_spam, text):
        """Train the system from user input"""
        assert isinstance(is_spam, bool)
        assert isinstance(text, unicode)

        """
        1. Create the features for text using create_features. Prepare a numpy
        matrix of one row as input to create_features.
        """
        this_X = np.array([(None, text, None)])
        feats = create_features(this_X, None)
       
        """
        2. Vectorize the feature dictionary using the deserialized
        DictVectorizer's transform function.
        """
        v = DictVectorizer(sparse=False)
        feats = v.fit_transform(feats)
        
        """
        3. We need to update self._x and self._y. Make sure they are lists and not
        None (None semantically means they are not yet initialized, used in
        script.py).
        """
        
        if self._x == None:
            self._x = []
        if self._y == None:
            self._y = []
        
        """
        4. Append the new data point to self._x and self._y. Make sure the types
        are consistent with that of loaded in script.py. self._x should be a list
        of lists of floats. self._y should be a list of ints.
        """
        self._x.append(feats)
        self._y.append(1 if is_spam else 0)
        
    def predict_input(self, text):
        """Returns True if TEXT is spam"""
        assert isinstance(text, unicode)
        
        """
        1. Create features for text using create_features (same as step 2 above).
        """
        this_X = np.array([(None, text, None)])
        feats = create_features(this_X, None)
        
        """
        2. Vectorize the feature dictionary using the deserialized
        DictVectorizer's transform function (same as step 2 above).
        """
        v = DictVectorizer(sparse=False)
        feats = v.fit_transform(feats)
        
        """
        3. Use self.predict(x) to predict for a vector x. The learning is done by
        periodically exchanging models in the background.
        """
        return self.predict(feats) == 1.0
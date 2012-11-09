from os import path

from Tribler.dispersy.database import Database
from Tribler.dispersy.revision import update_revision_information

if __debug__:
    from Tribler.dispersy.dprint import dprint

# update version information directly from SVN
update_revision_information("$HeadURL: https://svn.tribler.org/abc/branches/release-6.0.x-GOLF/release-6.0.x/Tribler/community/effort/database.py $", "$Revision: 28797 $")

LATEST_VERSION = 1

schema = u"""
CREATE TABLE input(
 id INTEGER,
 x STRING,
 y INTEGER
 );
 
CREATE TABLE option(key TEXT PRIMARY KEY, value BLOB);
INSERT INTO option(key, value) VALUES('database_version', '""" + str(LATEST_VERSION) + """');
 """

cleanup = u"""
DELETE FROM input;
"""

class GossipDatabase(Database):
    if __debug__:
        __doc__ = schema

    def __init__(self, dispersy):
        self._dispersy = dispersy
        super(GossipDatabase, self).__init__(path.join(dispersy.working_directory, u"sqlite", u"golf.db"))
        dispersy.database.attach_commit_callback(self.commit)

    def cleanup(self):
        self.executescript(cleanup)

    def check_database(self, database_version):
        assert isinstance(database_version, unicode)
        assert database_version.isdigit()
        assert int(database_version) >= 0
        database_version = int(database_version)

        # setup new database with current database_version
        if database_version < 1:
            self.executescript(schema)
            self.commit()

        return LATEST_VERSION
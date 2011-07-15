# written by Raynor Vliegendhart
# see LICENSE.txt for license information

import os
import sys
import wx
from Tribler.__init__ import LIBRARYNAME
from Tribler.Main.vwxGUI.list_body import ListItem, FixedListBody, NativeIcon
from Tribler.Main.vwxGUI.GuiUtility import GUIUtility
from Tribler.Main.vwxGUI.list import GenericSearchList
from Tribler.Main.vwxGUI.list_header import ListHeader
from Tribler.Main.vwxGUI.list_details import TorrentDetails
from Tribler.Main.vwxGUI.tribler_topButton import LinkStaticText

from __init__ import *
from traceback import print_exc

DEBUG = True

BUNDLE_FONT_SIZE_DECREMENT = 1 # TODO: on my machine this results in fontsize 7, a bit too small I think? 
BUNDLE_FONT_COLOR = (50,50,50)

BUNDLE_NUM_COLS = 3
BUNDLE_NUM_ROWS = 3

class BundleListItem(ListItem):
    
    def __init__(self, parent, parent_list, columns, data, original_data, leftSpacer = 0, rightSpacer = 0, showChange = False, list_selected = LIST_SELECTED):
        # fetch bundle and descriptions
        self.bundle = bundle = original_data['bundle']
        self.general_description = original_data.get('bundle_general_description')
        self.description = original_data.get('bundle_description')
        
        # use the head as original_data (needed for SearchList)
        original_data = bundle[0]
        
        # call the original constructor
        ListItem.__init__(self, parent, parent_list, columns, data, original_data, leftSpacer, rightSpacer, showChange, list_selected)
        
        # Now add the BundleListView (after AddComponents)
        self.AddBundlePanel()
        self.bundlepanel.Layout()
        
        self.expanded_panel = None
        self.expanded_panel_shown = False
        
    def AddBundlePanel(self):
        self.bundlepanel = BundlePanel(self, self.parent_list, self.bundle[1:], 
                                       self.general_description, self.description,
                                       -BUNDLE_FONT_SIZE_DECREMENT)
        self.AddEvents(self.bundlepanel)
        self.vSizer.Add(self.bundlepanel, 1, wx.EXPAND)
        
    def RefreshData(self, data):
        infohash, item_data, original_data = data
        if isinstance(original_data, dict) and 'bundle' in original_data:
            if DEBUG:
                print >>sys.stderr, "*** BundleListItem.RefreshData: bundle changed:", original_data['key'], '#1+%s' % (len(original_data['bundle'])-1)
            
            bundle = original_data['bundle']
            self.bundle = bundle
            
            if DEBUG:
                print >>sys.stderr, "*** BundleListItem.RefreshData: calling ListItem.RefreshData() with head"
            ListItem.RefreshData(self, (infohash, item_data, bundle[0]))
                        
            if DEBUG:
                print >>sys.stderr, "*** BundleListItem.RefreshData: calling BundlePanel.SetHits()"
            
            self.bundlepanel.SetHits(bundle[1:])
            self.bundlepanel.UpdateHeader(original_data['bundle_general_description'], original_data['bundle_description'])
            self.Highlight(1)
        else:
            self._RefreshDataNonBundle(data)
            
    def _RefreshDataNonBundle(self, data):
        infohash, item_data, original_data = data
        if DEBUG:
            print >>sys.stderr, "*** BundleListItem._RefreshDataNonBundle: single hit changed:", repr(item_data[0])
        
        if isinstance(original_data, dict):
            hit_to_update = None
            for hit in self.bundle:
                if hit['infohash'] == infohash:
                    hit_to_update = hit
                    break
            
            if hit_to_update:
                for k, v in original_data.iteritems():
                    hit_to_update[k] = v
            elif DEBUG:
                print >>sys.stderr, "*** BundleListItem._RefreshDataNonBundle: couldn't find hit in self.bundle!"
            
        elif DEBUG:
            print >>sys.stderr, "*** BundleListItem._RefreshDataNonBundle: data[2] != dict!"
        
        if infohash == self.bundle[0]['infohash']:
            if DEBUG:
                print >>sys.stderr, "*** BundleListItem._RefreshDataNonBundle: calling ListItem.RefreshData() with head"
            ListItem.RefreshData(self, data)
            
        else:
            self.bundlepanel.RefreshDataBundleList(infohash, original_data)
    
    def GetExpandedPanel(self):
        return self.expanded_panel

    def Expand(self, panel):
        # Similar to ListItem base class logic, except we insert the panel
        # to the vSizer at a specific index, instead of adding it to the end.
        if getattr(panel, 'SetCursor', False):
            panel.SetCursor(wx.StockCursor(wx.CURSOR_DEFAULT))
            #panel.SetFont(panel.GetDefaultAttributes().font)
        
        self.expanded_panel = panel
        self.ShowExpandedPanel()
    
    def Collapse(self):
        # Do most important part of base class logic first:
        self.expanded = False
        self.ShowSelected()
        
        # But grab the correct panel to return!
        panel_item = self.expanded_panel 
        self.expanded_panel = None
        self.expanded_panel_shown = False
        
        # Also collapse the bundlepanel
        self.bundlepanel.ChangeState(BundlePanel.COLLAPSED)
        return panel_item
    
    def OnClick(self, event):
        if not self.expanded or self.expanded_panel_shown:
            ListItem.OnClick(self, event)
        else:
            self.ShowExpandedPanel()
    
    def ShowExpandedPanel(self, show=True):
        panel = self.expanded_panel
        if panel is not None and show != self.expanded_panel_shown:
            if show:
                panel.Show()
                # Insert, instead of add:
                self.vSizer.Insert(1, panel, 0, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, 3)
                
                if getattr(self, 'expandedState', False):
                    self.expandedState.SetBitmap(self.GetIcon(self.list_selected, 1))
                
                self.button.Hide()
                
                # Only keep 1 panel open at all times, so close panels in the bundlepanel, if any:
                self.bundlepanel.CollapseExpandedItem()
            else:
                panel.Hide()
                self.vSizer.Remove(panel)
                
                if getattr(self, 'expandedState', False):
                    self.expandedState.SetBitmap(self.GetIcon(self.list_selected, 0))
                
                self.button.Show()
                
            self.parent_list.OnChange()
            self.expanded_panel_shown = show
            self.Layout()
    
    def BackgroundColor(self, color):
        ListItem.BackgroundColor(self, color)
        self.bundlepanel.SetBackgroundColour(color)

class BundlePanel(wx.Panel):
    
    COLLAPSED, PARTIAL, FULL = range(3)
    
    icons = None
    @classmethod
    def load_icons(cls):
        if not cls.icons:
            icons = cls.icons = {}
            guiUtility = GUIUtility.getInstance()
            utility = guiUtility.utility
            base_path = os.path.join(utility.getPath(), LIBRARYNAME, "Main", "vwxGUI", "images")
            
            icons['info'] = wx.Bitmap(os.path.join(base_path, "info.png"), wx.BITMAP_TYPE_ANY)
    
    def __init__(self, parent, parent_list, hits, general_description = None, description = None, font_increment=0):
        # preload icons
        self.load_icons()
        self.parent_listitem = parent
        self.parent_list = parent_list
        
        wx.Panel.__init__(self, parent)
        self.hits = hits
        self.state = BundlePanel.COLLAPSED
        
        self.general_description = general_description
        self.description = description
        
        self.font_increment = font_increment
        self.vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.SetBackgroundColour(wx.WHITE)
        
        self.AddHeader()
        self.AddGrid()
        
        self.SetSizer(self.vsizer)
    
    def AddHeader(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.header = wx.StaticText(self, -1, '')
        # Keep header font the same...
        # TODO: perhaps introduce two font_increment params in constructor?
        #font = self.header.GetFont()
        #font.SetPointSize(font.GetPointSize() + self.font_increment)
        #self.header.SetFont(font)
        
        self.info_icon = wx.StaticBitmap(self, -1, self.icons['info'])
        
        self.SetGeneralDescription(self.general_description)
        self.SetDescription(self.description)
        
        #sizer.Add(self.info_icon, 0)
        #sizer.Add(self.header, 0, wx.LEFT, 2)
        
        sizer.Add(self.header, 0, wx.RIGHT, 5)
        sizer.Add(self.info_icon, wx.TOP, 7)
        self.vsizer.Add(sizer, 0, wx.LEFT, 22)
    
    def UpdateHeader(self, general_description, description):
        self.general_description = general_description
        self.description = description
        self.SetGeneralDescription(general_description)
        self.SetDescription(description)
    
    def AddGrid(self):
        self.grid = wx.FlexGridSizer(BUNDLE_NUM_ROWS, BUNDLE_NUM_COLS, 0, 0)
        self.grid.SetFlexibleDirection(wx.HORIZONTAL)
        self.grid.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_NONE)
        self.grid.SetMinSize((1,-1))
        
        for i in xrange(BUNDLE_NUM_ROWS):
            self.grid.AddGrowableRow(i, 1)
        
        for j in xrange(BUNDLE_NUM_COLS):
            self.grid.AddGrowableCol(j, 1)
        
        self.UpdateGrid()
        self.vsizer.Add(self.grid, 1, wx.EXPAND | wx.LEFT, 30)
    
    def UpdateGrid(self):
        self.Freeze()
        self.grid.ShowItems(False)
        self.grid.Clear(deleteWindows = True)
        
        N = BUNDLE_NUM_ROWS * BUNDLE_NUM_COLS
        items_to_add = min(N, len(self.hits))
        if len(self.hits) > N:
            items_to_add -= 1
        
        for i in range(items_to_add):
            hit = self.hits[i] 

            new_text = LinkStaticText(self, hit['name'], icon = False, icon_type = 'tree', icon_align = wx.ALIGN_LEFT, font_increment = self.font_increment, font_colour = BUNDLE_FONT_COLOR)
            new_text.Bind(wx.EVT_LEFT_UP, self.OnBundleLinkClick)
            new_text.SetMinSize((1,-1))
            new_text.action = hit
            self.grid.Add(new_text, 0, wx.ALL | wx.EXPAND, 5)
            
        for i in range(BUNDLE_NUM_COLS - items_to_add):
            self.grid.AddSpacer((1,-1))
        
        if len(self.hits) > N:
            caption = '(%s more...)' % (len(self.hits) - N + 1)
            
            more_label = LinkStaticText(self, caption, icon = False, icon_align = wx.ALIGN_LEFT, font_increment = self.font_increment, font_colour = BUNDLE_FONT_COLOR)
            more_label.Bind(wx.EVT_LEFT_UP, self.OnMoreClick)
            self.grid.Add(more_label, 0, wx.ALL | wx.EXPAND, 5)
            
        self.parent_listitem.AddEvents(self.grid)
        
        if self.state != self.COLLAPSED:
            self.ShowGrid(False)
        self.Thaw()
    
    def ShowGrid(self, show):
        if show:
            self.grid.ShowItems(True)
            #self.vsizer.Add(self.grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 30)
        else:
            self.grid.ShowItems(False)
            #self.vsizer.Detach(self.grid)
    
    def UpdateList(self):
        bundlelist = getattr(self, 'bundlelist', None)
        if bundlelist:
            bundlelist.SetData(self.hits)
    
    def ShowList(self, show):
        bundlelist = getattr(self, 'bundlelist', None)
        if bundlelist is None and show:
            max_list = BUNDLE_NUM_ROWS * BUNDLE_NUM_COLS
            if len(self.hits) != BUNDLE_NUM_ROWS * BUNDLE_NUM_COLS:
                max_list -= 1
            
            self.bundlelist = BundleListView(parent = self, list_item_max = max_list)
            self.vsizer.Add(self.bundlelist, 0, wx.EXPAND | wx.LEFT, 20)
            
            # SetData does wx.Yield, which could cause a collapse event to be processed within the setdata
            # method
            wx.CallAfter(self.bundlelist.SetData, self.hits)
        
        elif bundlelist is not None and not show:
            self.vsizer.Detach(bundlelist)
            self.bundlelist.Show(False)
            self.bundlelist.Destroy()
            self.bundlelist = None
        
    def OnChange(self, scrollToTop = False):
        self.Layout()
        self.parent_listitem.Layout()
        self.parent_list.OnChange(scrollToTop)
    
    def CollapseExpandedItem(self):
        if self.state != BundlePanel.COLLAPSED:
            self.bundlelist.list.OnCollapse()
    
    def RefreshDataBundleList(self, key, data):
        bundlelist = getattr(self, 'bundlelist', None)
        if bundlelist is not None:
            bundlelist.RefreshData(key, data)
    
    def SetDescription(self, description):
        self.description = description
        self.header.SetToolTipString(description)
        self.info_icon.SetToolTipString(description)
    
    def SetGeneralDescription(self, general_description):
        if general_description:
            general_description = unicode(self.general_description)
        else:
            general_description = u'Similar'
        
        self.general_description = general_description
        self.header.SetLabel(u'%s items (%s):' % (general_description, len(self.hits)))
    
    def SetHits(self, hits):
        if self.hits != hits:
            self.hits = hits
            
            self.UpdateGrid()
            self.UpdateList()
            
            self.Layout()
    
    def ChangeState(self, new_state, doLayout=True):
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            if new_state == BundlePanel.COLLAPSED:
                self.ShowList(False)
                self.ShowGrid(True)
            else:
                if new_state == BundlePanel.PARTIAL or new_state == BundlePanel.FULL:
                    self.ShowGrid(False)
                    if old_state == BundlePanel.COLLAPSED:
                        self.ShowList(True)
                        
                    if new_state == BundlePanel.FULL:
                        self.bundlelist.OnLoadAll()

            if DEBUG:
                statestr = lambda st: ['COLLAPSED', 'PARTIAL', 'FULL'][st]
                print >>sys.stderr, '*** BundlePanel.ChangeState: %s --> %s' % (statestr(old_state), statestr(new_state))
                print >>sys.stderr, '\tName:', self.parent_listitem.bundle[0]['name']
            
    
    def ExpandAndScrollToHit(self, hit):
        id = hit['infohash']
        
        self.bundlelist.ExpandItem(id)
        #self.ScrollToId(id)
        self.parent_listitem.ShowSelected()
        
    def ScrollToId(self, id):
        parent_listitem_dy = self.parent_listitem.GetPosition()[1]
        self_dy = self.GetPosition()[1]
        hit_item_dy = self.bundlelist.VerticalItemOffset(id)
        
        total_y = parent_listitem_dy + self_dy + hit_item_dy
        
        ppu = self.parent_list.GetScrollPixelsPerUnit()[1]
        sy = total_y / ppu
        
        if DEBUG:
            print >>sys.stderr, \
            '*SCROLL*: p_li self hit (total) / ppu, sy:  %s %s %s (%s) / %s, %s' \
            % (parent_listitem_dy, self_dy, hit_item_dy, total_y, ppu, sy)
            sizer_h = self.parent_list.vSizer.GetSize()[1]
            print >>sys.stderr, '*SCROLL* parent_list vertical scroll height:', sizer_h/ppu
            # ^ This line confirms that we sometimes want to scroll beyond the size of the
            #   vsizer. Apparently the vsizer's size hasn't changed when we want to scroll...
        
        # ...therefore we should delay the scroll:
        #wx.CallAfter(self.parent_list.Scroll, -1, sy)
        wx.CallLater(100, self.parent_list.Scroll, -1, sy)
    
    def OnBundleLinkClick(self, event):
        listitem = self.GetParent()
        
        if not listitem.expanded:
            # Make sure the listitem is marked as expanded
            listitem.Freeze()
            listitem.OnClick(event)
            
            # but hide the panel
            listitem.ShowExpandedPanel(False)
            listitem.Thaw()
        
        staticText = event.GetEventObject()
        action = getattr(staticText, 'action', None)
        if action is not None:
            # Reason for non-persistence (for now) is least-surprise.
            # If the user collapses a bundled listitem, the previously 
            # clicked item is still at the same location.
            self.hits.remove(action)
            self.hits.insert(0, action)
        
            self.ChangeState(BundlePanel.PARTIAL)
            self.ExpandAndScrollToHit(action)
        
            
    def OnMoreClick(self, event):
        self.ChangeState(BundlePanel.FULL)
        
        event.Skip()
    
    def SetSelectedBundleLink(self, control=None):
        for bundletext in self.texts:
            bundletext.ShowSelected(bundletext == control)
            
    def SetBackgroundColour(self, colour):
        wx.Panel.SetBackgroundColour(self, colour)
        
        if getattr(self, 'grid', False):
            for sizeritem in self.grid.GetChildren():
                if sizeritem.IsWindow():
                    child = sizeritem.GetWindow()
                    if isinstance(child, wx.Panel):
                        child.SetBackgroundColour(colour)
    
class BundleListView(GenericSearchList):
    
    def __init__(self, parent = None, list_item_max = None):
        self.list_item_max = list_item_max
        columns = [{'name':'Name', 'width': wx.LIST_AUTOSIZE, 'sortAsc': True, 'icon': 'tree'}, \
                   {'name':'Size', 'width': '8em', 'style': wx.ALIGN_RIGHT, 'fmt': self.format_size, 'sizeCol': True}, \
                   {'type':'method', 'width': wx.LIST_AUTOSIZE_USEHEADER, 'method': self.CreateRatio, 'name':'Popularity'}, \
                   {'type':'method', 'width': -1, 'method': self.CreateDownloadButton}]
        
        GenericSearchList.__init__(self, columns, LIST_GREY, [7,7], True, parent=parent)
    
    def CreateHeader(self):
        # Normally, the column-widths are fixed during this phase
        # Or perhaps easier... just create the simplest header, but don't return it:
        header = ListHeader(self, self.columns)
        header.Destroy()
        
    def CreateFooter(self):
        pass 
    
    def CreateList(self):
        return ExpandableFixedListBody(self, self, self.columns, self.spacers[0], self.spacers[1], self.singleSelect, self.showChange, list_item_max = self.list_item_max)
    
    def OnExpand(self, item):
        # Keep only one panel open at all times, thus we make sure the parent is closed
        bundlepanel = self.parent
        bundlepanel.parent_listitem.ShowExpandedPanel(False)
        
        return BundleTorrentDetails(item, item.original_data)
    
    def OnCollapseInternal(self, item):
        pass
    
    def OnChange(self, scrollToTop = False):
        self.parent.OnChange(scrollToTop)
    
    def ExpandItem(self, id):
        # id == infohash
        self.list.Select(id, raise_event=True)
        
    def VerticalItemOffset(self, id):
        # id == infohash
        item = self.list.items[id]
        return item.GetPosition()[1]

class ExpandableFixedListBody(FixedListBody):
    
    def OnChange(self, scrollToTop = False):
        FixedListBody.OnChange(self, scrollToTop)
        
        self.parent_list.OnChange(scrollToTop)
    
class BundleTorrentDetails(TorrentDetails):
    def __init__(self, parent, torrent, compact=True):
        TorrentDetails.__init__(self, parent, torrent, compact=True)
    
    def _showTorrent(self, torrent, information):
        TorrentDetails._showTorrent(self, torrent, information)
        self.buttonPanel.Hide()
        self.details.Layout()
    
    def ShowPanel(self, *args, **kwargs):
        pass
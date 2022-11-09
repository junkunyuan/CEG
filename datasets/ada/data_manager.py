import dassl.data.data_manager as dmanager
from dassl.data.datasets import Datum

def split_path(path):
    if path.find('/') >= 0:
        return path.split('/')
    else:
        return path.split("\\")

class ActiveWrapper(dmanager.DatasetWrapper):
    def __init__(self, cfg, data_source, transform, is_train):
        super().__init__(cfg, data_source, transform=transform, is_train=is_train)
    
    def add_item(self, impath, label, domain):
        print(impath, label, domain)
        item = Datum(impath=impath, label=int(label), domain=int(domain))
        self.data_source.append(item)

    def add_items(self, items):
        for item in items:
            self.add_item(*item)

    def remove_item(self, impath, label, domain):
        item = Datum(impath=impath, label=int(label), domain=int(domain))
        print(self.data_source.index(item))
        self.data_source.remove(item)

    def remove_items(self, items):
        for item in items:
            self.remove_item(*item)

    def update_data_datum(self, data_source):
        self.data_source = data_source
    
    def update_data_items(self, items):
        data_source = []
        for item in items:
            data_source.append(Datum(impath=item[0], label=int(item[1]), domain=item[2]))
        self.data_source = data_source

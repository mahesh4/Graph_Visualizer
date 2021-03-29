import sys
import numpy as np
from app.skyline.instance_object import InstanceObject


class TimelineObject:
    def __init__(self, value_list):
        self.instances = [InstanceObject(value) for value in value_list]
        self.instances_count = len(self.instances)
        self.prob_lower_limit = 0
        self.prob_upper_limit = 1
        self.dim = len(value_list[0])
        self.layers = list()
        self.UPmax = None
        self.UPmin = 1  # Visited(Processed) Subset Pmin
        self.virtual_instance_min = None
        self.virtual_instance_max = None

    def check_dominating(self, dominating_instance, dominated_instance):
        """

        :param dominating_instance:
        :param dominated_instance:
        :return:
        """
        less_count, equal_count, greater_count = 0, 0, 0
        for i in range(self.dim):
            if dominating_instance.value[i] < dominated_instance.value[i]:
                less_count += 1
            elif dominating_instance.value[i] > dominated_instance.values[i]:
                greater_count += 1
            else:
                equal_count += 1

        if greater_count == 0 and less_count > 0:
            return 1
        elif less_count == 0 and greater_count > 0:
            return -1
        else:
            return 0

    def find_layer(self, instance):
        """

        :param instance:
        :return:
        """
        beg = 0
        end = len(self.layers)
        while beg <= end:
            mid = int(beg + (end - beg) / 2)
            update = False
            for layer_instance in self.layers[mid]:
                dominate = self.check_dominating(instance, layer_instance)
                if dominate == 1:
                    end = mid - 1
                    update = True
                    break
                elif dominate == -1:
                    beg = mid + 1
                    update = True
                    break
            # End of loop
            if not update:
                return mid

        return 0 if beg == 0 else beg + 1

    def partition_instances_to_layers(self):
        self.instances.sort(key=lambda x: np.sum(x.value))
        for idx, instance in enumerate(self.instances):
            layer_index = self.find_layer(instance)
            if layer_index < len(self.layers):
                self.layers[layer_index].append(idx)
            elif layer_index == len(self.layers):
                self.layers.append([idx])
            else:
                raise Exception("wrong layer found")
        # End of loop

    def is_last_instance_in_layer(self, instance_idx):
        layer_idx = self.find_layer(self.instances[instance_idx])
        return True, layer_idx if self.layers[layer_idx][-1] == instance_idx else False, layer_idx

    def update_Pmax(self, layer_idx):
        self.UPmax = np.max([self.instances[instance_id].probability for instance_id in self.layers[layer_idx]])

    def find_next_instance_in_layer(self, layer_idx, instance_index):
        for index, layer_instance_idx in enumerate(self.layers[layer_idx]):
            if layer_instance_idx == instance_index:
                if index + 1 == len(self.layers[layer_idx]):
                    if layer_idx + 1 < len(self.layers):
                        return self.layers[layer_idx + 1][0]
                    else:
                        return None
                else:
                    return self.layers[layer_idx][index + 1]
        # End of loop
        return None

    def update_UPmin(self, probability):
        pass

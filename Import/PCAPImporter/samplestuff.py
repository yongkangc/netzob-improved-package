def get_packet(self):
        """Returns the raw representation of this packet and its
        children as a string. The output from this method is a packet
        ready to be transmitted over the wire.
        """
        self.calculate_checksum()

        data = self.get_data_as_string()
        if data:
            return self.get_buffer_as_string() + data
        else:
            return self.get_buffer_as_string()

def get_data_as_string(self):
        "Returns all data from children of this header as string"

        if self.child():
            return self.child().get_packet()
        else:
            return None


def get_ACK(self):
        return self.get_flag(16)

def get_flag(self, bit):
        if self.get_th_flags() & bit:
            return 1
        else:
            return 0

def get_th_flags(self):
        return self.get_word(12) & self.TCP_FLAGS_MASK

def get_word(self, index, order='!'):
        "Return 2-byte word at 'index'. See struct module's documentation to understand the meaning of 'order'."
        index = self.__validate_index(index, 2)
        if -2 == index:
            bytes = self.__bytes[index:]
        else:
            bytes = self.__bytes[index:index + 2]
        (value, ) = struct.unpack(order + 'H', bytes.tostring())
        return value
def get_icmp_type(self):
        return self.get_byte(0)
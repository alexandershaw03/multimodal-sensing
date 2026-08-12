from cortex import Cortex
from pylsl import StreamInfo, StreamOutlet


class Subcribe():
    """
    Subscribe to Cortex data streams and publish raw EEG to LSL.
    """

    def __init__(self, app_client_id, app_client_secret, **kwargs):
        """
        Construct Cortex client, bind callbacks, and create ATS EEG LSL outlet.
        """

        print("Subscribe __init__")

        # ---------------------------------------------------------
        # CORTEX CONNECTION
        # ---------------------------------------------------------

        self.c = Cortex(
            app_client_id,
            app_client_secret,
            debug_mode=True,
            **kwargs
        )

        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_eeg_data=self.on_new_eeg_data)
        self.c.bind(new_mot_data=self.on_new_mot_data)
        self.c.bind(new_dev_data=self.on_new_dev_data)
        self.c.bind(new_met_data=self.on_new_met_data)
        self.c.bind(new_pow_data=self.on_new_pow_data)
        self.c.bind(inform_error=self.on_inform_error)

        # ---------------------------------------------------------
        # ATS EEG LSL OUTLET
        # ---------------------------------------------------------

        self.eeg_info = StreamInfo(
            name='ATS_EEG_RAW',
            type='EEG',
            channel_count=5,
            nominal_srate=128,
            channel_format='float32',
            source_id='emotiv_insight_ats'
        )

        channels = self.eeg_info.desc().append_child('channels')

        for label in ['AF3', 'T7', 'Pz', 'T8', 'AF4']:
            channel = channels.append_child('channel')
            channel.append_child_value('label', label)
            channel.append_child_value('unit', 'microvolts')
            channel.append_child_value('type', 'EEG')

        self.eeg_info.desc().append_child_value(
            'manufacturer',
            'EMOTIV'
        )

        self.eeg_info.desc().append_child_value(
            'model',
            'Insight'
        )

        self.eeg_outlet = StreamOutlet(self.eeg_info)

        print()
        print("========================================")
        print(" ATS EEG LSL OUTLET CREATED")
        print(" Stream: ATS_EEG_RAW")
        print(" Channels: AF3, T7, Pz, T8, AF4")
        print(" Sample rate: 128 Hz")
        print("========================================")
        print()

    def start(self, streams, headset_id=''):
        """
        Start Cortex subscription process.
        """

        self.streams = streams

        if headset_id != '':
            self.c.set_wanted_headset(headset_id)

        self.c.open()

    def sub(self, streams):
        """
        Subscribe to Cortex streams.
        """

        self.c.sub_request(streams)

    def unsub(self, streams):
        """
        Unsubscribe from Cortex streams.
        """

        self.c.unsub_request(streams)

    def on_new_data_labels(self, *args, **kwargs):
        """
        Handle labels returned by Cortex.
        """

        data = kwargs.get('data')

        if data is None:
            return

        stream_name = data['streamName']
        stream_labels = data['labels']

        print(
            '{} labels are : {}'.format(
                stream_name,
                stream_labels
            )
        )

    def on_new_eeg_data(self, *args, **kwargs):
        """
        Receive raw EEG from Cortex and publish the five electrode channels to ATS_EEG_RAW via LSL.

        Cortex EEG layout for Insight:

        0 = COUNTER
        1 = INTERPOLATED
        2 = AF3
        3 = T7
        4 = Pz
        5 = T8
        6 = AF4
        7 = RAW_CQ
        8 = MARKER_HARDWARE
        """

        data = kwargs.get('data')

        if data is None:
            return

        eeg_data = data.get('eeg')

        if eeg_data is None:
            return

        # Make sure Cortex returned enough values
        if len(eeg_data) < 7:
            return

        sample = [
            float(eeg_data[2]),   # AF3
            float(eeg_data[3]),   # T7
            float(eeg_data[4]),   # Pz
            float(eeg_data[5]),   # T8
            float(eeg_data[6])    # AF4
        ]

        # Push one 5-channel EEG sample into LSL
        self.eeg_outlet.push_sample(sample)

    def on_new_mot_data(self, *args, **kwargs):
        """
        Handle motion data.
        """

        data = kwargs.get('data')

        if data is not None:
            print('motion data: {}'.format(data))

    def on_new_dev_data(self, *args, **kwargs):
        """
        Handle device-information data.
        """

        data = kwargs.get('data')

        if data is not None:
            print('dev data: {}'.format(data))

    def on_new_met_data(self, *args, **kwargs):
        """
        Handle performance-metrics data.
        """

        data = kwargs.get('data')

        if data is not None:
            print('pm data: {}'.format(data))

    def on_new_pow_data(self, *args, **kwargs):
        """
        Handle band-power data.
        """

        data = kwargs.get('data')

        if data is not None:
            print('pow data: {}'.format(data))

    def on_create_session_done(self, *args, **kwargs):
        """
        Subscribe once Cortex session has been created.
        """

        print('on_create_session_done')

        self.sub(self.streams)

    def on_inform_error(self, *args, **kwargs):
        """
        Handle Cortex errors.
        """

        error_data = kwargs.get('error_data')

        print(error_data)


def main():

    # ---------------------------------------------------------
    # CORTEX CREDENTIALS
    # ---------------------------------------------------------

    # Replace "xxx"'s below with actual client ID and secret (same as ATS EEG TECH)
    your_app_client_id = ' xxx '
    your_app_client_secret = ' xxx '

    # Create ATS Cortex -> LSL bridge
    s = Subcribe(
        your_app_client_id,
        your_app_client_secret
    )

    # Raw EEG only for now
    streams = ['eeg']

    s.start(streams)


if __name__ == '__main__':
    main()

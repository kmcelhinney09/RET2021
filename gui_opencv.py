import PySimpleGUI as sg
import cv2
import numpy as np
import random


celebdict = {'Abigail Spencer': 'lookslike/abigailspencer.png',
             'Alan Cummings': 'lookslike/alancummings.png',
             'Alec Baldwin': 'lookslike/alecbaldwin.png',
             'Alex Newell': 'lookslike/alexnewell.png',
             'Alexa davalos': 'lookslike/alexadavalos.png',
             'Aliana Lohan': 'lookslike/alianalohan.png',
             'Alicia Keys': 'lookslike/aliciakeys.png',
             'Alicia Silverstone': 'lookslike/aliciasilverstone.png',
             'Alicia Vikander': 'lookslike/aliciavikander.png',
             'Allen Leech': 'lookslike/allenleech.png',
             'Amanda Peet': 'lookslike/amandapeet.png',
             'Amber Heard': 'lookslike/amberheard.png',
             'Amber Riley': 'lookslike/amberriley.png',
             'America Ferrera': 'lookslike/americaferrera.png',
             'Amy Mccarthy': 'lookslike/amymccarthy.png',
             'Amy Sedaris': 'lookslike/amysedaris.png',
             'Ana Gasteyer': 'lookslike/anagasteyer.png',
             'Angela Kinsey': 'lookslike/angelakinsey.png',
             'Angell Conwell': 'lookslike/angellconwell.png',
             'Annabelle Wallis': 'lookslike/annabellewallis.png',
             'Anne Hathaway': 'lookslike/annehathaway.png',
             'Ansel Elgort': 'lookslike/anselelgort.png',
             'Arthur Wahlberg': 'lookslike/arthurwahlberg.png',
             'Ashlee Simpson': 'lookslike/ashleesimpson.png',
             'Ashley Graham': 'lookslike/ashleygraham.png',
             'Ashley Olsen': 'lookslike/ashleyolsen.png',
             'Austin Butler': 'lookslike/austinbutler.png',
             'Ava Phillippe': 'lookslike/avaphillippe.png',
             'Barack Obama ': 'lookslike/barackobama.png',
             'Bella Hadid': 'lookslike/bellahadid.png',
             'Ben Affleck': 'lookslike/benaffleck.png',
             'Ben Mckenzie': 'lookslike/benmckenzie.png',
             'Benicio Del Toro': 'lookslike/beniciodeltoro.png',
             'Bette Midler': 'lookslike/bettemidler.png',
             'Beyonce': 'lookslike/beyonce.png',
             'Billie Eilish': 'lookslike/billieeilish.png',
             'Bono': 'lookslike/bono.png',
             'Brad Pitt': 'lookslike/bradpitt.png',
             'Brandon Jenner': 'lookslike/brandonjenner.png',
             'Brianna Cuoco': 'lookslike/briannacuoco.png',
             'Bridget Regan': 'lookslike/bridgetregan.png',
             'Britney Spears': 'lookslike/britneyspears.png',
             'Brittany Murphy': 'lookslike/brittanymurphy.png',
             'Brody Jenner': 'lookslike/brodyjenner.png',
             'Bruce Darnell': 'lookslike/brucedarnell.png',
             'Bryce Dallas Howard': 'lookslike/brycedallashoward.png',
             'Cameron Diaz': 'lookslike/camerondiaz.png',
             'Camille Leblanc-Bazinet': 'lookslike/camilleleblanc-bazinet.png',
             'Cara Delevigne': 'lookslike/caradelevigne.png',
             'Carla Bruni': 'lookslike/carlabruni.png',
             'Carlos Bardem': 'lookslike/carlosbardem.png',
             'Carrie Underwood': 'lookslike/carrieunderwood.png',
             'Chad Smith': 'lookslike/chadsmith.png',
             'Gavin Degraw': 'lookslike/gavindegraw.png',
             'Georgia May Jagger': 'lookslike/georgiamayjagger.png',
             'Gigi Hadid': 'lookslike/gigihadid.png',
             'Ginnifer Goodwin': 'lookslike/ginnifergoodwin.png',
             'Grace Gummer': 'lookslike/gracegummer.png',
             'Guillaume Canet': 'lookslike/guillaumecanet.png',
             'Haley Duff': 'lookslike/haleyduff.png',
             'Hannah Simone': 'lookslike/hannahsimone.png',
             'Heath Ledger': 'lookslike/heathledger.png',
             'Helen Hunt': 'lookslike/helenhunt.png',
             'Helen Mirren Young': 'lookslike/helenmirrenyoung.png',
             'Henry Cavill': 'lookslike/henrycavill.png',
             'Hilary': 'lookslike/hilary.png',
             'Hilary Swank': 'lookslike/hilaryswank.png',
             'Hugh Jackman': 'lookslike/hughjackman.png',
             'Ian Somerhalder': 'lookslike/iansomerhalder.png',
             'Ilham Anas': 'lookslike/ilhamanas.png',
             'Isaac Mizrahi': 'lookslike/isaacmizrahi.png',
             'J. Alexander': 'lookslike/jalexander.png',
             'JJ Abrams': 'lookslike/jjabrams.png',
             'JWoww': 'lookslike/jwoww.png',
             'Jackie Chan': 'lookslike/jackiechan.png',
             'Jada Pinkett Smith': 'lookslike/jadapinkettsmith.png',
             'Jaden Smith': 'lookslike/jadensmith.png',
             'Jai Courtney': 'lookslike/jaicourtney.png',
             'Jaime King': 'lookslike/jaimeking.png',
             'Jaime Pressly': 'lookslike/jaimepressly.png',
             'Jameela Jamil': 'lookslike/jameelajamil.png',
             'James Franco': 'lookslike/jamesfranco.png',
             'James Lackey': 'lookslike/jameslackey.png',
             'James Wahlberg': 'lookslike/jameswahlberg.png',
             'Jamie Lynn Spears': 'lookslike/jamielynnspears.png',
             'Jana Kramer': 'lookslike/janakramer.png',
             'Jane Levy': 'lookslike/janelevy.png',
             'Jane Lynch': 'lookslike/janelynch.png',
             'Jason Segel': 'lookslike/jasonsegel.png',
             'Javier Bardem': 'lookslike/javierbardem.png',
             'Jeff Bridges': 'lookslike/jeffbridges.png',
             'Jeff Daniels': 'lookslike/jeffdaniels.png',
             'Jeffrey Dean Morgan': 'lookslike/jeffreydeanmorgan.png',
             'Jeffrey Tambor': 'lookslike/jeffreytambor.png',
             'Jennifer Connelly': 'lookslike/jenniferconnelly.png',
             'Jennifer Coolidge': 'lookslike/jennifercoolidge.png',
             'Jennifer Garner': 'lookslike/jennifergarner.png',
             'Jennifer Lawrence': 'lookslike/jenniferlawrence.png',
             'Jennifer Morrison': 'lookslike/jennifermorrison.png',
             'Jenny Mccarthy': 'lookslike/jennymccarthy.png',
             'Jeremy Irons': 'lookslike/jeremyirons.png',
             'Jesse Plemons': 'lookslike/jesseplemons.png',
             'Jessica Alba': 'lookslike/jessicaalba.png',
             'Jessica Biel': 'lookslike/jessicabiel.png',
             'Jessica Chastain': 'lookslike/jessicachastain.png',
             'Jessica Simpson': 'lookslike/jessicasimpson.png',
             'Jim Carrey': 'lookslike/jimcarrey.png',
             'Joe Jonas': 'lookslike/joejonas.png',
             'Joey Lauren Adams': 'lookslike/joeylaurenadams.png',
             'John Mayer': 'lookslike/johnmayer.png',
             'Jon Stewart Richard Lewis': 'lookslike/jonstewartrichardlewis.png',
             'Jonathan Pryce': 'lookslike/jonathanpryce.png',
             'Jonathan Rhys Meyers': 'lookslike/jonathanrhysmeyers.png',
             'Jordan Hinson': 'lookslike/jordanhinson.png',
             'Jordin Sparks': 'lookslike/jordinsparks.png',
             'Joseph Fiennes': 'lookslike/josephfiennes.png',
             'Joseph Gordon Levitt': 'lookslike/josephgordonlevitt.png',
             'Josh Duhamel': 'lookslike/joshduhamel.png',
             'Josh McRoberts': 'lookslike/joshmcroberts.png',
             'Julia Stiles': 'lookslike/juliastiles.png',
             'Julie Bowen': 'lookslike/juliebowen.png',
             'Justin Bartha': 'lookslike/justinbartha.png',
             'Kaley Cuoco': 'lookslike/kaleycuoco.png',
             'Kate Mara': 'lookslike/katemara.png',
             'Kate Middleton': 'lookslike/katemiddleton.png',
             'Kate Walsh': 'lookslike/katewalsh.png',
             'Kathryn Hahn': 'lookslike/kathrynhahn.png',
             'Katy Perry': 'lookslike/katyperry.png',
             'Keira Knightley': 'lookslike/keiraknightley.png',
             'Kelly Lynch': 'lookslike/kellylynch.png',
             'Kendall Jenner': 'lookslike/kendalljenner.png',
             'Kenny G': 'lookslike/kennyg.png',
             'Kevin Jonas': 'lookslike/kevinjonas.png',
             'Khloe Kardashian': 'lookslike/khloekardashian.png',
             'Kiernan Shipka': 'lookslike/kiernanshipka.png',
             'Kim Kardashian West': 'lookslike/kimkardashianwest.png',
             'Kofi Annan': 'lookslike/kofiannan.png',
             'Kourtney Kardashian': 'lookslike/kourtneykardashian.png',
             'Kris Humphries': 'lookslike/krishumphries.png',
             'Kris Jenner': 'lookslike/krisjenner.png',
             'Kristen Stewart': 'lookslike/kristenstewart.png',
             'Kristen Wilson': 'lookslike/kristenwilson.png',
             'Krysten Ritter': 'lookslike/krystenritter.png',
             'Kurt Russel': 'lookslike/kurtrussel.png',
             'Kylie Jenner': 'lookslike/kyliejenner.png',
             'Kyra Sedgwick': 'lookslike/kyrasedgwick.png',
             'Lake Bell': 'lookslike/lakebell.png',
             'Lara Stone': 'lookslike/larastone.png',
             'Laura Benanti': 'lookslike/laurabenanti.png',
             'Lauren Conrad': 'lookslike/laurenconrad.png',
             'Leelee Sobieski': 'lookslike/leeleesobieski.png',
             'Leighton Meester': 'lookslike/leightonmeester.png',
             'Liam Hemsworth': 'lookslike/liamhemsworth.png',
             'Lili Reinhart': 'lookslike/lilireinhart.png',
             'Lily Collins': 'lookslike/lilycollins.png',
             'Lindsay Lohan': 'lookslike/lindsaylohan.png',
             'Lisa Bonet': 'lookslike/lisabonet.png',
             'Logan Marshall Green': 'lookslike/loganmarshallgreen.png',
             'Luke Wilson': 'lookslike/lukewilson.png',
             'Mamie Gummer': 'lookslike/mamiegummer.png',
             'Margot Robbie': 'lookslike/margotrobbie.png',
             'Mark Tomlin': 'lookslike/marktomlin.png',
             'Mark Wahlberg': 'lookslike/markwahlberg.png',
             'Mary-Kate Olsen': 'lookslike/mary-kateolsen.png',
             'Matt Bomer': 'lookslike/mattbomer.png',
             'Matt Damon': 'lookslike/mattdamon.png',
             'Megan Fox': 'lookslike/meganfox.png',
             'Melania Trump': 'lookslike/melaniatrump.png',
             'Michael Sheen': 'lookslike/michaelsheen.png',
             'Mickey Rourke': 'lookslike/mickeyrourke.png',
             'Mila Kunis': 'lookslike/milakunis.png',
             'Miley Cyrus': 'lookslike/mileycyrus.png',
             'Mimi Rogers': 'lookslike/mimirogers.png',
             'Minka Kelly': 'lookslike/minkakelly.png',
             'Missy Peregrym': 'lookslike/missyperegrym.png',
             'Monica Cruz': 'lookslike/monicacruz.png',
             'Morgan Freeman': 'lookslike/morganfreeman.png',
             'Morris Chestnut': 'lookslike/morrischestnut.png',
             'Natalie Portman': 'lookslike/natalieportman.png',
             'Neels Visser': 'lookslike/neelsvisser.png',
             'Nelly Furtado': 'lookslike/nellyfurtado.png',
             'Nick Jonas': 'lookslike/nickjonas.png',
             'Nicole Hilton': 'lookslike/nicolehilton.png',
             'Nicole Scherzinger': 'lookslike/nicolescherzinger.png',
             'Nina Dobrev': 'lookslike/ninadobrev.png',
             'Noah Cyrus': 'lookslike/noahcyrus.png',
             'Nora Zehetner': 'lookslike/norazehetner.png',
             'Omar Epps': 'lookslike/omarepps.png',
             'Owen Wilson': 'lookslike/owenwilson.png',
             'Paris Hilton': 'lookslike/parishilton.png',
             'Patricia Arquette': 'lookslike/patriciaarquette.png',
             'Patrick Dempsey': 'lookslike/patrickdempsey.png',
             'Patrick Swayze': 'lookslike/patrickswayze.png',
             'Paul Giamatti': 'lookslike/paulgiamatti.png',
             'Paul Reubens': 'lookslike/paulreubens.png',
             'Paul Wahlberg': 'lookslike/paulwahlberg.png',
             'Paz Vega': 'lookslike/pazvega.png',
             'Penelope Cruz': 'lookslike/penelopecruz.png',
             'Penn Badgley': 'lookslike/pennbadgley.png',
             'Peter Jackson': 'lookslike/peterjackson.png',
             'Phillip Phillips': 'lookslike/phillipphillips.png',
             'Pippa Middleton': 'lookslike/pippamiddleton.png',
             'Pope Francis': 'lookslike/popefrancis.png',
             'Portia de Rossi': 'lookslike/portiaderossi.png',
             'Priscilla Faia': 'lookslike/priscillafaia.png',
             'Rachel Bilson': 'lookslike/rachelbilson.png',
             'Rachel McAdams': 'lookslike/rachelmcadams.png',
             'Ralph Fiennes': 'lookslike/ralphfiennes.png',
             'Ray Romano': 'lookslike/rayromano.png',
             'Renee Zellweger': 'lookslike/reneezellweger.png',
             'Richard Hammond': 'lookslike/richardhammond.png',
             'Rob Lowe': 'lookslike/roblowe.png',
             'Robert Wahlberg': 'lookslike/robertwahlberg.png',
             'Robin Williams': 'lookslike/robinwilliams.png',
             'Ronda Rousey': 'lookslike/rondarousey.png',
             'Rooney Mara': 'lookslike/rooneymara.png',
             'Roselyn Sanchez': 'lookslike/roselynsanchez.png',
             'Russell Crowe': 'lookslike/russellcrowe.png',
             'Sanjay Gupta': 'lookslike/sanjaygupta.png',
             'Sarah Hyland': 'lookslike/sarahhyland.png',
             'Scarlett Johansson': 'lookslike/scarlettjohansson.png',
             'Selena Gomez': 'lookslike/selenagomez.png',
             'Selma Blair': 'lookslike/selmablair.png',
             'Serena Williams': 'lookslike/serenawilliams.png',
             'Seth MacFarlane': 'lookslike/sethmacfarlane.png',
             'Shia LaBeouf': 'lookslike/shialabeouf.png',
             'Skrillex': 'lookslike/skrillex.png',
             'Skylar Astin': 'lookslike/skylarastin.png',
             'Solange': 'lookslike/solange.png',
             'Sophie von Haselberg': 'lookslike/sophievonhaselberg.png',
             'Stephen Baldwin': 'lookslike/stephenbaldwin.png',
             'Stephen Colbert': 'lookslike/stephencolbert.png',
             'Taylor Lautner': 'lookslike/taylorlautner.png',
             'Terry Notary': 'lookslike/terrynotary.png',
             'Thandie Newton': 'lookslike/thandienewton.png',
             'Tim Robbins': 'lookslike/timrobbins.png',
             'Timothy Olyphant': 'lookslike/timothyolyphant.png',
             'Tom Hardy': 'lookslike/tomhardy.png',
             'Tony Hayward': 'lookslike/tonyhayward.png',
             'Tracy Morgan': 'lookslike/tracymorgan.png',
             'Tyler the Creator': 'lookslike/tylerthecreator.png',
             'Usher Raymond': 'lookslike/usherraymond.png',
             'Venus Williams': 'lookslike/venuswilliams.png',
             'Victoria Justice': 'lookslike/victoriajustice.png',
             'Warren Elgort': 'lookslike/warrenelgort.png',
             'Weird Al': 'lookslike/weirdal.png',
             'Wendie Malick': 'lookslike/wendiemalick.png',
             'Will Ferrell': 'lookslike/willferrell.png',
             'William Baldwin': 'lookslike/williambaldwin.png',
             'William Dafoe': 'lookslike/williamdafoe.png',
             'Willow Smith': 'lookslike/willowsmith.png',
             'Zach Braff': 'lookslike/zachbraff.png',
             'Zachary Quinto': 'lookslike/zacharyquinto.png',
             'Zoe Saldana': 'lookslike/zoesaldana.png',
             'Zooey Deschanel': 'lookslike/zooeydeschanel.png',
             'Zooey Kravitz': 'lookslike/zooeykravitz.png',
             'alisa Allapach': 'lookslike/alisaallapach.png',
             'arden cho': 'lookslike/ardencho.png',
             'augusta xu-holland': 'lookslike/augustaxu-holland.png',
             'brenda song': 'lookslike/brendasong.png',
             'camila cabello': 'lookslike/camilacabello.png'}


def main():
    sg.theme('BrownBlue')

    def random_celeb():
        celeb_names = list(celebdict)
        x = random.choice(celeb_names)

        print(x)
        print(celebdict[x])
        window['image'].update(filename=celebdict[x])
        window['celebname'].update(x)
        window['probabilitylist'].update(celebdict[x])
        window.bind('<s>', 'Snap')
        window.bind('<e>', 'Exit')

    # define the window layout
    videofeed = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
                 [sg.Image(filename='', key='video'), sg.Image(filename='', key='snap')],
                 [sg.Button('Snap', size=(10, 1), font='Any 14'),
                  sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]
    message = [
        [sg.Text('You looklike', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Image(filename='lookslike/tomhardy.png', key='image')],
        [sg.Text('', key='celebname', size=(40, 1), justification='center', font='Helvetica 20')]
    ]
    #######third colum box
    third = [
        [sg.Text('Probability', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Multiline(default_text='', size=(55, 33), key='probabilitylist')],

    ]

    # third column box#########

    layout = [
        [
            sg.Column(videofeed, element_justification="c"),
            sg.VSeperator(),
            sg.Column(message, element_justification="c"),
            sg.VSeperator(),
            sg.Column(third, element_justification="c")  ######## this is the third column

        ]
    ]
    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400), return_keyboard_events=True)

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #

    recording = False
    cap = cv2.VideoCapture(0)
    recording = True
    while True:

        event, values = window.read(timeout=20)

        # ret, frame = cap.read()
        # video_start = cv2.imencode('.png', frame)[1].tobytes()
        # window['video'].update(data=video_start)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Snap':
            # recording = False
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
            cv2.imwrite('snapshot.png', frame)
            window['snap'].update(filename='snapshot.png')
            random_celeb()

        if recording:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['video'].update(data=imgbytes)


main()

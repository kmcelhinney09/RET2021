import PySimpleGUI as sg
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from operator import itemgetter
import pickle as pkl

# Face Net github repository: https://github.com/timesler/facenet-pytorch
# To use this code you are required to run the following command in the terminal in your conda environment:
# pip install facenet-pytorch

# Initiate the deep learning models
# Resnet is the model actually used for getting new representations of the input images
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet.classify = True

# Mtcnn is used for identifying where the face is in the input image and cropping the image
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)

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
             'Angelina Jolie': 'lookslike/angelinajolie.png',
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
             'Carmen Electra': 'lookslike/carmenelectra.png',
             'Carrie Underwood': 'lookslike/carrieunderwood.png',
             'Casey Affleck': 'lookslike/caseyaffleck.png',
             'Chace Crawford': 'lookslike/chacecrawford.png',
             'Chad Smith': 'lookslike/chadsmith.png',
             'Chandra Wilson': 'lookslike/chandrawilson.png',
             'Charlie Day': 'lookslike/charlieday.png',
             'Charlie Murphy': 'lookslike/charliemurphy.png',
             'Chimene Diaz': 'lookslike/chimenediaz.png',
             'Chord Overstreet': 'lookslike/chordoverstreet.png',
             'Chris Hemsworth': 'lookslike/chrishemsworth.png',
             'Chris O\'Donnell': 'lookslike/chrisodonnell.png',
             'Chrissy Teigen': 'lookslike/chrissyteigen.png',
             'Christian Bale': 'lookslike/christianbale.png',
             'Christopher Knight': 'lookslike/christopherknight.png',
             'Christopher Meloni': 'lookslike/christophermeloni.png',
             'Christopher Williams': 'lookslike/christopherwilliams.png',
             'Christy Turlington': 'lookslike/christyturlington.png',
             'Cindy Crawford': 'lookslike/cindycrawford.png',
             'Corey Feldman': 'lookslike/coreyfeldman.png',
             'Courteney Cox': 'lookslike/courteneycox.png',
             'DB Woodside': 'lookslike/dbwoodside.png',
             'DJ Qualls': 'lookslike/djqualls.png',
             'Dakota Fanning': 'lookslike/dakotafanning.png',
             'Dana Delany': 'lookslike/danadelany.png',
             'Dane Cook': 'lookslike/danecook.png',
             'Daniel Baldwin': 'lookslike/danielbaldwin.png',
             'Daniel Day Lewis': 'lookslike/danieldaylewis.png',
             'Daniel Radcliffe': 'lookslike/danielradcliffe.png',
             'Daryl Hannah': 'lookslike/darylhannah.png',
             'Dave Coulier': 'lookslike/davecoulier.png',
             'Dave Franco': 'lookslike/davefranco.png',
             'David Cross': 'lookslike/davidcross.png',
             'David Fumero': 'lookslike/davidfumero.png',
             'David Tennant': 'lookslike/davidtennant.png',
             'Deborah Cox': 'lookslike/deborahcox.png',
             'Demi Moore': 'lookslike/demimoore.png',
             'Denis leary': 'lookslike/denisleary.png',
             'Dianna Agron': 'lookslike/diannaagron.png',
             'Don Swayze': 'lookslike/donswayze.png',
             'Donald Glover': 'lookslike/donaldglover.png',
             'Doug E. Fresh': 'lookslike/dougefresh.png',
             'Drea de Matteo': 'lookslike/dreadematteo.png',
             'Ed Westwick': 'lookslike/edwestwick.png',
             'Eddie Murphy': 'lookslike/eddiemurphy.png',
             'Edie Falco': 'lookslike/ediefalco.png',
             'Elias Koteas': 'lookslike/eliaskoteas.png',
             'Eliza Coupe': 'lookslike/elizacoupe.png',
             'Eliza dushku': 'lookslike/elizadushku.png',
             'Elizabeth Banks': 'lookslike/elizabethbanks.png',
             'Elizabeth Olsen': 'lookslike/elizabetholsen.png',
             'Elle Fanning': 'lookslike/ellefanning.png',
             'Ellen Barkin': 'lookslike/ellenbarkin.png',
             'Emilie de Ravin': 'lookslike/emiliederavin.png',
             'Emily Deschanel': 'lookslike/emilydeschanel.png',
             'Emily Hand': 'Lookslike/emilyhand.png',
             'Emily Kinney': 'lookslike/emilykinney.png',
             'Emily Osment': 'lookslike/emilyosment.png',
             'Emma Stone': 'lookslike/emmastone.png',
             'Emma Watson': 'lookslike/emmawatson.png',
             'Emmanuelle Chriqui': 'lookslike/emmanuellechriqui.png',
             'Eric Stoltz': 'lookslike/ericstoltz.png',
             'Eva Mendes': 'lookslike/evamendes.png',
             'Evanna Lynch': 'lookslike/evannalynch.png',
             'Evelyne Brochu': 'lookslike/evelynebrochu.png',
             'Ezra Miller': 'lookslike/ezramiller.png',
             'Frankie Jonas': 'lookslike/frankiejonas.png',
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
             'Stephen Amell': 'lookslike/stephenamell',
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
             'camila cabello': 'lookslike/camilacabello.png',
             'constance wu': 'lookslike/constancewu.png',
             'ellen page': 'lookslike/ellenpage.png',
             'emma appleton': 'lookslike/emmaappleton.png'}
pickle_file = open("Saved_Classifiers.pkl", 'rb')
classifer_dict = pkl.load(pickle_file)
pickle_file.close()


def image_compare(input_image, window):
    image_match_probability = []

    image = Image.open(input_image)

    image_cropped = mtcnn(image, save_path="./snapshot_crop.png")

    # Get the processed representation for the image
    try:
        image_representation = resnet(image_cropped.unsqueeze(0)).detach().cpu()

        # Flatten the representation and convert it to a numpy array for saving
        image_representation_flat = image_representation.squeeze().numpy()
        image_representation_flat = image_representation_flat.reshape(1, -1)
    except:
        return (None, None)

    for name, each_classifier in classifer_dict.items():
        name_1 = name[0]
        name_2 = name[1]

        probabilities = each_classifier.predict_proba(image_representation_flat)
        image_match_probability.append([name_1, name_2, probabilities[0][1]])
    match_list = []
    max_prob = 0
    celeb_name = ''
    for index, match in enumerate(image_match_probability):
        if match[2] < .5:
            proba = 1 - match[2]
            celeb_set = 2
        else:
            proba = match[2]
            celeb_set = 1
        if proba > max_prob:
            max_prob = proba
            if celeb_set == 1:
                celeb_name = match[0]
            else:
                celeb_name = match[1]
        if celeb_set == 1:
            match_list.append([match[0], proba])
        else:
            match_list.append([match[1], proba])

        image_match_probability[index][2] = proba

        # print(celeb_name,proba)
    match_list = sorted(match_list, key=itemgetter(1), reverse=True)[:15]
    matches_to_send = ''
    for celeb_match in match_list:
        to_append = "You have {:.5f} probability of looking like {}".format(celeb_match[1], celeb_match[0])
        matches_to_send = matches_to_send + to_append + "\n" + "\n"
    return celeb_name, matches_to_send


def celeb_image_match(window, celeb_name_prediction, match_probabilites):
    window['snapshot'].update(filename="snapshot_crop.png")
    window['image'].update(filename=celebdict[celeb_name_prediction])
    window['celebname'].update(celeb_name_prediction,font='gothic 20')
    window['probabilitylist'].update(match_probabilites)


def main():
    sg.theme('BrownBlue')

    # define the window layout
    videofeed = [[sg.Text('UNR RECNT Doppelganger Demo', size=(40, 1), justification='center', font='gothic 20')],
                 [sg.Image(filename='', key='video'), sg.Image(filename='', key='snapshot')],
                 [sg.Button('Snap', size=(10, 1), font='Any 14'),
                  sg.Button('Clear', size=(10,1), font='Any 14'),
                  sg.Button('Exit', size=(10, 1), font='Any 14'), ]]
    message = [
        [sg.Text('You looklike', size=(20, 1), justification='center', font='gothic 20')],
        [sg.Image(filename='', key='image')],
        [sg.Text('', key='celebname', size=(30, 1), justification='center', font='gothic 20')]
    ]
    #######third colum box
    third = [
        [sg.Text('Probability', size=(20, 1), justification='center', font='gothic 20')],
        [sg.Multiline(default_text='', size=(75, 33), key='probabilitylist')],

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
                       layout, location=(800, 400), finalize=True)
    window.bind('<s>', 'Snap')
    window.bind('<e>', 'Exit')
    window.bind('<c>', 'Clear')

    recording = False
    cap = cv2.VideoCapture(0)
    recording = True
    while True:

        event, values = window.read(timeout=20)

        ret, frame = cap.read()
        video_start = cv2.imencode('.png', frame)[1].tobytes()
        window['video'].update(data=video_start)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Snap':
            # recording = False
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
            cv2.imwrite('snapshot.png', frame)
            window['snapshot'].update(filename='snapshot.png')
            window['image'].update(filename='')

            celeb_name_prediction, match_probabilities = image_compare('snapshot.png', window)
            if celeb_name_prediction == None:
                window['celebname'].update("Snap must be of a face",font='gothic 10')
                window['image'].update(filename='')
            else:
                celeb_image_match(window, celeb_name_prediction, match_probabilities)
        elif event == "Clear":
            window['celebname'].update('')
            window['image'].update(filename='')
            window['snapshot'].update(filename='')
            window['probabilitylist'].update('')

        if recording:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['video'].update(data=imgbytes)



main()

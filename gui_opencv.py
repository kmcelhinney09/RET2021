import PySimpleGUI as sg
import cv2
import numpy as np

"""
Demo program that displays a webcam using OpenCV
"""
testdict = {
    "Ben Affleck": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/benaffleck.png",
    "Beyonce": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/beyonce.png",
    "Brianna Cuoco": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/briannacuoco.png",
    "Casey Affleck": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/caseyaffleck.png",
    "Dave Franco": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/davefranco.png",
    "Haley Duff": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/haleyduff.png",
    "Hilary": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/hilary.png",
    "James Franco": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/jamesfranco.png",
    "Kaley Cuoco": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/kaleycuoco.png",
    "Solange": "/home/teacheraccount/PycharmProjects/perceptron1/lookslike/solange.png",
}
celebdict = {"Abigail Spencer": "lookslike/abigailspencer.png", "Alan Cumming": "lookslike/alancumming.png",
             "Alec Baldwin": "lookslike/alecbaldwin.png", "Alex Newell": "lookslike/alexnewell.png",
             "Alexa davalos": "lookslike/alexadavalos.png", "Aliana Lohan": "lookslike/alianalohan.png",
             "Alicia Keys": "lookslike/aliciakeys.png", "Alicia Silverstone": "lookslike/aliciasilverstone.png",
             "Alicia Vikander": "lookslike/aliciavikander.png", "alisa Allapach": "lookslike/alisaallapach.png",
             "Allen Leech": "lookslike/allenleech.png", "Amanda Peet": "lookslike/amandapeet.png",
             "Amber Heard": "lookslike/amberheard.png", "Amber Riley": "lookslike/amberriley.png",
             "America Ferrera": "lookslike/americaferrera.png", "Amy Mccarthy": "lookslike/amymccarthy.png",
             "Amy Sedaris": "lookslike/amysedaris.png", "Ana Gasteyer": "lookslike/anagasteyer.png",
             "Angela Kinsey": "lookslike/angelakinsey.png", "Angelina Jolie": "lookslike/angelinajolie.png",
             "angelina jolie": "lookslike/angelinajolie.png", "Angell Conwell": "lookslike/angellconwell.png",
             "Annabelle Wallis": "lookslike/annabellewallis.png", "Anne Hathaway": "lookslike/annehathaway.png",
             "Ansel Elgort": "lookslike/anselelgort.png", "Anya Chalotra": "lookslike/anyachalotra.png",
             "arden cho": "lookslike/ardencho.png", "Arthur Wahlberg": "lookslike/arthurwahlberg.png",
             "Ashlee Simpson": "lookslike/ashleesimpson.png", "Ashley Graham": "lookslike/ashleygraham.png",
             "Ashley Olson": "lookslike/ashleyolson.png", "augusta xu-holland": "lookslike/augustaxu-holland.png",
             "Austin Butler": "lookslike/austinbutler.png", "Ava Phillippe": "lookslike/avaphillippe.png",
             "Barak Obama": "lookslike/barakobama.png", "Bella Hadid": "lookslike/bellahadid.png",
             "Ben Affleck": "lookslike/benaffleck.png", "Ben Feldman": "lookslike/benfeldman.png",
             "ben mckenzie": "lookslike/benmckenzie.png", "Benicio Del Toro": "lookslike/beniciodeltoro.png",
             "Bette Midler": "lookslike/bettemidler.png", "Beyonce": "lookslike/beyonce.png",
             "Billie Eilish": "lookslike/billieeilish.png", "Bob Saget": "lookslike/bobsaget.png",
             "Bono": "lookslike/bono.png", "Brad Pitt": "lookslike/bradpitt.png",
             "Brandon Jenner": "lookslike/brandonjenner.png", "brenda song": "lookslike/brendasong.png",
             "Brianna Cuoco": "lookslike/briannacuoco.png", "Bridget Regan": "lookslike/bridgetregan.png",
             "Brittany Murphy": "lookslike/brittanymurphy.png", "Brittany Spears": "lookslike/brittanyspears.png",
             "Brody Jenner": "lookslike/brodyjenner.png", "Bruce Darnell": "lookslike/brucedarnell.png",
             "Bryce Dallas Howard": "lookslike/brycedallashoward.png", "Burn Gorman": "lookslike/burngorman.png",
             "Cameron Diaz": "lookslike/camerondiaz.png", "camila cabello": "lookslike/camilacabello.png",
             "camille leblanc-bazinet": "lookslike/camilleleblanc-bazinet.png",
             "Cara Delevigne": "lookslike/caradelevigne.png", "Carl Mayer": "lookslike/carlmayer.png",
             "carla bruni": "lookslike/carlabruni.png", "Carlos Bardem": "lookslike/carlosbardem.png",
             "Carmen Electra": "lookslike/carmenelectra.png", "Carrie Underwood": "lookslike/carrieunderwood.png",
             "Casey Affleck": "lookslike/caseyaffleck.png", "Chace Crawford": "lookslike/chacecrawford.png",
             "Chad Smith": "lookslike/chadsmith.png", "Chandra Wilson": "lookslike/chandrawilson.png",
             "Charley Murphey": "lookslike/charleymurphey.png", "Charlie Day": "lookslike/charlieday.png",
             "Chimene Diaz": "lookslike/chimenediaz.png", "Chord Overstreet": "lookslike/chordoverstreet.png",
             "Chris Hemsworth": "lookslike/chrishemsworth.png", "Chrissy Teigen": "lookslike/chrissyteigen.png",
             "Christian Bale": "lookslike/christianbale.png", "Christopher Knight": "lookslike/christopherknight.png",
             "Christopher Meloni": "lookslike/christophermeloni.png",
             "Christopher Williams": "lookslike/christopherwilliams.png",
             "Christy Burlington": "lookslike/christyburlington.png", "Cindy Crawford": "lookslike/cindycrawford.png",
             "Clint Eastwood": "lookslike/clinteastwood.png", "constance wu": "lookslike/constancewu.png",
             "Corey Feldman": "lookslike/coreyfeldman.png", "Courteney Cox": "lookslike/courteneycox.png",
             "Dakota Fanning": "lookslike/dakotafanning.png", "Dana Delany": "lookslike/danadelany.png",
             "Dane Cook": "lookslike/danecook.png", "Daniel Baldwin": "lookslike/danielbaldwin.png",
             "Daniel Day-Lewis": "lookslike/danielday-lewis.png", "Daniel Radcliffe": "lookslike/danielradcliffe.png",
             "Daryl Hannah": "lookslike/darylhannah.png", "dave coulier": "lookslike/davecoulier.png",
             "Dave Franco": "lookslike/davefranco.png", "Dave Gahan": "lookslike/davegahan.png",
             "David Cross": "lookslike/davidcross.png", "David Fumero": "lookslike/davidfumero.png",
             "David Tennant": "lookslike/davidtennant.png", "Dax Shephard": "lookslike/daxshephard.png",
             "DB Woodside": "lookslike/dbwoodside.png", "Deborah Cox": "lookslike/deborahcox.png",
             "Demi Moore": "lookslike/demimoore.png", "Dennis leary": "lookslike/dennisleary.png",
             "Dianna Agron": "lookslike/diannaagron.png", "Don Swaze": "lookslike/donswaze.png",
             "Donald Glover": "lookslike/donaldglover.png", "Doug E. Fresh": "lookslike/douge.fresh.png",
             "Dr Phil": "lookslike/drphil.png", "Drea de Matteo": "lookslike/dreadematteo.png",
             "Ed Westwick": "lookslike/edwestwick.png", "Eddie Murphy": "lookslike/eddiemurphy.png",
             "Edie Falco": "lookslike/ediefalco.png", "Eli Roth": "lookslike/eliroth.png",
             "Elias Koteas": "lookslike/eliaskoteas.png", "Elijah Wood": "lookslike/elijahwood.png",
             "Eliza Coupe": "lookslike/elizacoupe.png", "Eliza dushku": "lookslike/elizadushku.png",
             "Eliza Taylor": "lookslike/elizataylor.png", "Elizabeth Banks": "lookslike/elizabethbanks.png",
             "Elizabeth Olson": "lookslike/elizabetholson.png", "Elizabeth Reaser": "lookslike/elizabethreaser.png",
             "Elle Fanning": "lookslike/ellefanning.png", "Ellen Barkin": "lookslike/ellenbarkin.png",
             "ellen page": "lookslike/ellenpage.png", "Emilie de Ravin": "lookslike/emiliederavin.png",
             "Emily Deschanel": "lookslike/emilydeschanel.png", "Emily Kinney": "lookslike/emilykinney.png",
             "Emily Osment": "lookslike/emilyosment.png", "emma appleton": "lookslike/emmaappleton.png",
             "Emma Stone": "lookslike/emmastone.png", "Emma Watson": "lookslike/emmawatson.png",
             "Emmanuelle Chriqui": "lookslike/emmanuellechriqui.png", "Eric Stoltz": "lookslike/ericstoltz.png",
             "Eugena Washington": "lookslike/eugenawashington.png", "Eva Mendes": "lookslike/evamendes.png",
             "Evanna Lynch": "lookslike/evannalynch.png", "Frankie Jonas": "lookslike/frankiejonas.png",
             "Gavin Degraw": "lookslike/gavindegraw.png", "Gemma Chan": "lookslike/gemmachan.png",
             "Geoffrey Dean Morgan": "lookslike/geoffreydeanmorgan.png",
             "George Clooney": "lookslike/georgeclooney.png", "Georgia May Jagger": "lookslike/georgiamayjagger.png",
             "Gigi Hadid": "lookslike/gigihadid.png", "Ginnifer Goodwin": "lookslike/ginnifergoodwin.png",
             "Goldie Hawn": "lookslike/goldiehawn.png", "Grace Gummer": "lookslike/gracegummer.png",
             "grace huang": "lookslike/gracehuang.png", "Greta lee": "lookslike/gretalee.png",
             "guillaume canet": "lookslike/guillaumecanet.png", "Guillermo Zapata": "lookslike/guillermozapata.png",
             "Guy Pearce": "lookslike/guypearce.png", "Haley Duff": "lookslike/haleyduff.png",
             "Hannah Simone": "lookslike/hannahsimone.png", "Heath Ledger": "lookslike/heathledger.png",
             "Helen Hunt": "lookslike/helenhunt.png", "Helen Mirin Young": "lookslike/helenmirinyoung.png",
             "Henry Cavill": "lookslike/henrycavill.png", "Hilary": "lookslike/hilary.png",
             "Hilary Swank": "lookslike/hilaryswank.png", "Hugh Jackman": "lookslike/hughjackman.png",
             "Ian Somerhalder": "lookslike/iansomerhalder.png", "Ice Cube": "lookslike/icecube.png",
             "Iddo Golberg": "lookslike/iddogolberg.png", "Ilham Anas": "lookslike/ilhamanas.png",
             "India Love": "lookslike/indialove.png", "Isaac Mizrahi": "lookslike/isaacmizrahi.png",
             "J. Alexander": "lookslike/j.alexander.png", "Jackie Chan": "lookslike/jackiechan.png",
             "Jada Pinkett Smith": "lookslike/jadapinkettsmith.png", "Jaden Smith": "lookslike/jadensmith.png",
             "Jai Courtney": "lookslike/jaicourtney.png", "Jaime Pressly": "lookslike/jaimepressly.png",
             "Jameel a Jamil": "lookslike/jameelajamil.png", "James Franco": "lookslike/jamesfranco.png",
             "James Lackey": "lookslike/jameslackey.png", "James Patrick Stuart": "lookslike/jamespatrickstuart.png",
             "James Wahlberg": "lookslike/jameswahlberg.png", "Jamie King": "lookslike/jamieking.png",
             "Jamie Lynn Spears": "lookslike/jamielynnspears.png", "Jana Kramer": "lookslike/janakramer.png",
             "Jane Levy": "lookslike/janelevy.png", "Jane Lynch": "lookslike/janelynch.png",
             "Janelle Monae": "lookslike/janellemonae.png", "Jason Segel": "lookslike/jasonsegel.png",
             "Javier Bardem": "lookslike/javierbardem.png", "jeannie mai": "lookslike/jeanniemai.png",
             "Jeff Bridges": "lookslike/jeffbridges.png", "jeff daniels": "lookslike/jeffdaniels.png",
             "Jeffrey Tambor": "lookslike/jeffreytambor.png", "Jennifer Connelly": "lookslike/jenniferconnelly.png",
             "Jennifer Coolidge": "lookslike/jennifercoolidge.png", "Jennifer Garner": "lookslike/jennifergarner.png",
             "Jennifer Hudson": "lookslike/jenniferhudson.png", "Jennifer Lawrence": "lookslike/jenniferlawrence.png",
             "Jennifer Morrison": "lookslike/jennifermorrison.png", "Jenny Mccarthy": "lookslike/jennymccarthy.png",
             "Jeremy Irons": "lookslike/jeremyirons.png", "Jesse Plemons": "lookslike/jesseplemons.png",
             "Jessica Alba": "lookslike/jessicaalba.png", "Jessica Biel": "lookslike/jessicabiel.png",
             "Jessica Chastain": "lookslike/jessicachastain.png", "Jessica Simpson": "lookslike/jessicasimpson.png",
             "Jessica White": "lookslike/jessicawhite.png", "Jim Carrey": "lookslike/jimcarrey.png",
             "JJ Abrams": "lookslike/jjabrams.png", "Joe Jonas": "lookslike/joejonas.png",
             "Joey Lauren Adams": "lookslike/joeylaurenadams.png", "John Mayer": "lookslike/johnmayer.png",
             "Jon Stewart": "lookslike/jonstewart.png", "Jonathan Pryce": "lookslike/jonathanpryce.png",
             "Jonathan Rhys Meyers": "lookslike/jonathanrhysmeyers.png", "Jordan Hinson": "lookslike/jordanhinson.png",
             "Jordin Sparks": "lookslike/jordinsparks.png", "Joseph Fiennes": "lookslike/josephfiennes.png",
             "Joseph Gorden Levitt": "lookslike/josephgordenlevitt.png",
             "Josephine Langford": "lookslike/josephinelangford.png", "Josh Duhamel": "lookslike/joshduhamel.png",
             "Josh McRoberts": "lookslike/joshmcroberts.png", "Julia Stiles": "lookslike/juliastiles.png",
             "Julie Bowen": "lookslike/juliebowen.png", "Justin Bartha": "lookslike/justinbartha.png",
             "JWoww": "lookslike/jwoww.png", "Kaia Gerber": "lookslike/kaiagerber.png",
             "Kaley Cuoco": "lookslike/kaleycuoco.png", "Karrueche": "lookslike/karrueche.png",
             "Kate Hudson": "lookslike/katehudson.png", "Kate Mara": "lookslike/katemara.png",
             "Kate Middleton": "lookslike/katemiddleton.png", "Kate Walsh": "lookslike/katewalsh.png",
             "Kathryn Hahn": "lookslike/kathrynhahn.png", "Katy Perry": "lookslike/katyperry.png",
             "Keira Knightley": "lookslike/keiraknightley.png", "Kelly Lynch": "lookslike/kellylynch.png",
             "Kenny G": "lookslike/kennyg.png", "Kevin Alejandro": "lookslike/kevinalejandro.png",
             "Kevin Jonas": "lookslike/kevinjonas.png", "Khloe Kardashian": "lookslike/khloekardashian.png",
             "Kiernan Shipka": "lookslike/kiernanshipka.png", "Kim Kardashian West": "lookslike/kimkardashianwest.png",
             "Kirsten Storms": "lookslike/kirstenstorms.png", "Kofi Anan": "lookslike/kofianan.png",
             "Kourtney Kardashian": "lookslike/kourtneykardashian.png", "Kris Humphries": "lookslike/krishumphries.png",
             "Kris Jenner": "lookslike/krisjenner.png", "Kristen Stewart": "lookslike/kristenstewart.png",
             "kristen wilson": "lookslike/kristenwilson.png", "kristin kreuk": "lookslike/kristinkreuk.png",
             "Krysten Ritter": "lookslike/krystenritter.png", "Kurt Russel": "lookslike/kurtrussel.png",
             "kyra sedgwick": "lookslike/kyrasedgwick.png", "Lake Bell": "lookslike/lakebell.png",
             "Lance Reddick": "lookslike/lancereddick.png", "Lara Stone": "lookslike/larastone.png",
             "laura benanti": "lookslike/laurabenanti.png", "Lauren Conrad": "lookslike/laurenconrad.png",
             "Leelee Sobieski": "lookslike/leeleesobieski.png", "Leighton Meester": "lookslike/leightonmeester.png",
             "Liam Hemsworth": "lookslike/liamhemsworth.png", "Lili Reinhart": "lookslike/lilireinhart.png",
             "Lily allen": "lookslike/lilyallen.png", "Lily Collins": "lookslike/lilycollins.png",
             "Lindsay Lohan": "lookslike/lindsaylohan.png", "Lisa Bonet": "lookslike/lisabonet.png",
             "Logan Marshall Green": "lookslike/loganmarshallgreen.png", "Lucas Bravo": "lookslike/lucasbravo.png",
             "Luke Wilson": "lookslike/lukewilson.png", "Luke Youngblood": "lookslike/lukeyoungblood.png",
             "Mamie Gummer": "lookslike/mamiegummer.png", "Margarita Levieva": "lookslike/margaritalevieva.png",
             "Margot Robbie": "lookslike/margotrobbie.png", "marguerite moreau": "lookslike/margueritemoreau.png",
             "Mark Tomlin": "lookslike/marktomlin.png", "Mark Wahlberg": "lookslike/markwahlberg.png",
             "Mary-Kay Olson": "lookslike/mary-kayolson.png", "matt barnes": "lookslike/mattbarnes.png",
             "Matt Bomer": "lookslike/mattbomer.png", "Matt Damon": "lookslike/mattdamon.png",
             "Megan Fox": "lookslike/meganfox.png", "melania trump": "lookslike/melaniatrump.png",
             "Michael J Fox": "lookslike/michaeljfox.png", "Michael Sheen": "lookslike/michaelsheen.png",
             "Michael Weston": "lookslike/michaelweston.png", "Mickey Rourke": "lookslike/mickeyrourke.png",
             "Mila Kunis": "lookslike/milakunis.png", "Miley Cyrus": "lookslike/mileycyrus.png",
             "Mimi Rogers": "lookslike/mimirogers.png", "Minka Kelly": "lookslike/minkakelly.png",
             "Missy Peregrym": "lookslike/missyperegrym.png", "Monica Cruz": "lookslike/monicacruz.png",
             "Morgan Freeman": "lookslike/morganfreeman.png", "Morris Chestnut": "lookslike/morrischestnut.png",
             "Natalia lafourcade": "lookslike/natalialafourcade.png", "Natalie Portman": "lookslike/natalieportman.png",
             "Natasha Leggero": "lookslike/natashaleggero.png", "Neels Visser": "lookslike/neelsvisser.png",
             "Nelly Furtado": "lookslike/nellyfurtado.png", "Niall Horan": "lookslike/niallhoran.png",
             "Nichole Bloom": "lookslike/nicholebloom.png", "Nick Jonas": "lookslike/nickjonas.png",
             "Nicole Ari Parker": "lookslike/nicoleariparker.png",
             "Nicole Hilton": "lookslike/nicolehilton.png", "Nicole Scherzinger": "lookslike/nicolescherzinger.png",
             "Nina Dobrev": "lookslike/ninadobrev.png", "Niv Sultan": "lookslike/nivsultan.png",
             "Noah Cyrus": "lookslike/noahcyrus.png", "Nora Zehetner": "lookslike/norazehetner.png",
             "Oliver Hudson": "lookslike/oliverhudson.png", "Olivia Munn": "lookslike/oliviamunn.png",
             "Omar Epps": "lookslike/omarepps.png", "OShea Jackson Jr": "lookslike/osheajacksonjr.png",
             "Owen Wilson": "lookslike/owenwilson.png", "Paris Berelc": "lookslike/parisberelc.png",
             "Paris Hilton": "lookslike/parishilton.png", "Patricia Arquette": "lookslike/patriciaarquette.png",
             "Patrick": "lookslike/patrick.png", "Patrick Dempsey": "lookslike/patrickdempsey.png",
             "Paul Giamatti": "lookslike/paulgiamatti.png", "Paul Wahlberg": "lookslike/paulwahlberg.png",
             "Paz Vega": "lookslike/pazvega.png", "Pee Wee Herman": "lookslike/peeweeherman.png",
             "Penelope": "lookslike/penelope.png", "Penelope Cruz": "lookslike/penelopecruz.png",
             "Penn Badgley": "lookslike/pennbadgley.png", "Peter Jackson": "lookslike/peterjackson.png",
             "Phillip Phillips": "lookslike/phillipphillips.png", "Pippa Middleton": "lookslike/pippamiddleton.png",
             "Pope Francis": "lookslike/popefrancis.png", "Portia de Rossi": "lookslike/portiaderossi.png",
             "Priscilla Faia": "lookslike/priscillafaia.png", "Pyper America": "lookslike/pyperamerica.png",
             "Rachel Bilson": "lookslike/rachelbilson.png", "Rachel McAdams": "lookslike/rachelmcadams.png",
             "Ralph Fiennes": "lookslike/ralphfiennes.png", "Rasheeda Frost": "lookslike/rasheedafrost.png",
             "Raven Goodwin": "lookslike/ravengoodwin.png", "Ray Romano": "lookslike/rayromano.png",
             "Reese Witherspoon": "lookslike/reesewitherspoon.png", "Reid Scott": "lookslike/reidscott.png",
             "Renee Zellweger": "lookslike/reneezellweger.png", "Richard Hammond": "lookslike/richardhammond.png",
             "Richard Lewis": "lookslike/richardlewis.png", "Rihanna": "lookslike/rihanna.png",
             "Rita Ora": "lookslike/ritaora.png", "Rob Lowe": "lookslike/roblowe.png",
             "Robert Wahlberg": "lookslike/robertwahlberg.png", "Robin Williams": "lookslike/robinwilliams.png",
             "Ronda Rousey": "lookslike/rondarousey.png", "Rooney Mara": "lookslike/rooneymara.png",
             "Roselyn Sanchez": "lookslike/roselynsanchez.png", "Ruby Rose": "lookslike/rubyrose.png",
             "Rumor Willis": "lookslike/rumorwillis.png", "russell crowe": "lookslike/russellcrowe.png",
             "Sanjay Gupta": "lookslike/sanjaygupta.png", "Sarah Hyland": "lookslike/sarahhyland.png",
             "Scarlett Johansson": "lookslike/scarlettjohansson.png", "Selena Gomez": "lookslike/selenagomez.png",
             "Selma Blair": "lookslike/selmablair.png", "Serena Williams": "lookslike/serenawilliams.png",
             "Serinda Swan": "lookslike/serindaswan.png", "Seth MacFarlane": "lookslike/sethmacfarlane.png",
             "Sharlto Copley": "lookslike/sharltocopley.png", "Shia LeBeouf": "lookslike/shialebeouf.png",
             "Skrillex": "lookslike/skrillex.png", "Skylar Astin": "lookslike/skylarastin.png",
             "Solange": "lookslike/solange.png", "Sophie von Haselberg": "lookslike/sophievonhaselberg.png",
             "Stana Katic": "lookslike/stanakatic.png", "Stephen Baldwin": "lookslike/stephenbaldwin.png",
             "Stephen Colbert": "lookslike/stephencolbert.png", "tatiana maslany": "lookslike/tatianamaslany.png",
             "Taylor Lautner": "lookslike/taylorlautner.png", "Terry Notary": "lookslike/terrynotary.png",
             "Thandie Newton": "lookslike/thandienewton.png", "Tim Robbins": "lookslike/timrobbins.png",
             "Timothy Olyphant": "lookslike/timothyolyphant.png", "Tina Majorino": "lookslike/tinamajorino.png",
             "Tom Hardy": "lookslike/tomhardy.png", "Tony Hayward": "lookslike/tonyhayward.png",
             "Tracy Morgan": "lookslike/tracymorgan.png", "Tyler the Creator": "lookslike/tylerthecreator.png",
             "Usher Raymond": "lookslike/usherraymond.png", "Venus": "lookslike/venus.png",
             "Victoria Justice": "lookslike/victoriajustice.png", "Vinessa Shaw": "lookslike/vinessashaw.png",
             "Warren Elgort": "lookslike/warrenelgort.png", "Weird Al": "lookslike/weirdal.png",
             "Wendie Malick": "lookslike/wendiemalick.png", "Will Ferrell": "lookslike/willferrell.png",
             "Willem Dafoe": "lookslike/willemdafoe.png", "William Baldwin": "lookslike/williambaldwin.png",
             "William Dafoe": "lookslike/williamdafoe.png", "Willow Smith": "lookslike/willowsmith.png",
             "y serkis": "lookslike/yserkis.png", "Zach Braff": "lookslike/zachbraff.png",
             "Zachary Quinto": "lookslike/zacharyquinto.png", "Zoe Saldana": "lookslike/zoesaldana.png",
             "zoey deutch": "lookslike/zoeydeutch.png", "Zooey Deschanel": "lookslike/zooeydeschanel.png",
             "Zooey Kravitz": "lookslike/zooeykravitz.png", }


def main():
    sg.theme('BrownBlue')

    # define the window layout
    videofeed = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
                 [sg.Image(filename='', key='video'), sg.Image(filename='', key='snap')],
                 [sg.Button('Snap', size=(10, 1), font='Any 14'),
                  sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]
    message = [
        [sg.Text('You looklike', size=(40, 1), justification='center', font='Helvetica 20')],
        [sg.Image(filename=fname, key='image')]
    ]
    layout = [
        [
            sg.Column(videofeed, element_justification="c"),
            sg.VSeperator(),
            sg.Column(message, element_justification="c"),
        ]
    ]
    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

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

        if recording:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['video'].update(data=imgbytes)


main()

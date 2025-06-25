robocup-ssl.github.io Rules of the RoboCup Small Size League 2025-02-22 60–76
minutes References to the male gender in the rules with respect to referees,
team members, officials, etc. are for simplification and apply to both males and
females. 1. League Overview 1.1. Committees The Small Size League (like every
other league of the RoboCup) is run by close cooperation of three different
committees (executive committee, technical committee and organizing committee),
all with a different set of responsibilities. The members of the respective
committees can be found on the official RoboCup Small Size League website
(https://ssl.robocup.org). In practice, there is no strict separation between
the technical and the organizing committee. Members of both committees often
work together on the joint set of tasks. Additionally, the members of the local
organizing committee organize the RoboCup event for all leagues. 1.1.1.
Executive Committee Executive committee members are responsible for the long
term goals of the Small Size League and thus have also contact to other leagues
as well as to the RoboCup federation. The executive committee presents the Small
Size League and its achievements to the RoboCup federation every year and gets
feedback to organize the league. Executive committee members are elected by the
board of trustees of the RoboCup federation. They serve 3-year terms. 1.1.2.
Technical Committee The technical committee of the Small Size League is
responsible for the technical aspects of the RoboCup, such as maintaining the
rules and the shared software. All members are elected by the team leaders of
the teams which have participated in the previous competition. 1.1.3. Organizing
Committee The organizing committee of the Small Size League is responsible for
preparing and organizing the competition. This mainly includes making the
schedule, performing the qualification process, and running the competition. The
committee members are selected by the executive committee of the league and the
RoboCup trustees. 1.1.4. Local Organizing Committee The local organizing
committee is responsible for planning and executing the event itself in
accordance with the needs of the different leagues. This includes setting up the
team areas (fields, network, tables, whiteboard, screens, etc.), creating a
schedule for the event and implementing a safety and security concept. 1.2.
Divisions The Small Size League is divided into two divisions with separate
tournaments, namely division A and division B. Division A is aimed at advanced
teams whereas new and/or less competitive teams can play in division B. Each
team will only play in one of those two divisions. When submitting the
qualification material, the team also chooses a preferred division including a
short rationale. The members of the organizing committee will have the final
word. Information about the qualification process can be found on the official
RoboCup Small Size League website (https://ssl.robocup.org). A summary of
differences between the two divisions can be found in the Appendix. Divisions
allow for more radical advancements in the Small Size League without drastically
raising the entry barrier for new teams. Additionally, they also considerably
increase the amount of matches between teams of similar skill. 2. Playing
Environment 2.1. Field Setup 2.1.1. Dimensions The field of play must be
rectangular and of the following size: Division A: 13.4 meters times 10.4 meters
with a playing area of 12 meters times 9 meters Division B: 10.4 meters times
7.4 meters with a playing area of 9 meters times 6 meters The exact field
dimensions and the field markings at the venue may vary by up to ±10% in each
linear dimension. The two figures below show the dimensions of the field, the
goals and special field areas, measured in millimeters between the line centers.
Figure 1 shows the dimensions for division A and figure 2 for division B. The
numbers in the figures below show the distances in millimeters between the line
centers. quad size field Figure 1. Field dimensions and markings for division A
double size field Figure 2. Field dimensions and markings for division B 2.1.2.
Field Surface The playing surface is green felt mat or carpet. The floor under
the carpet is level, flat, and hard. The field surface will continue for 0.7
meters beyond the field lines on all sides. The outer 0.4 meters of this runoff
area, separated from the robot area by a 0.1 meters tall black wall, is used as
a designated walking area for the referee and the assistant referee. The
remaining 0.3 meters are the field margins. 2.1.3. Field Markings The field of
play is marked with lines. All lines are 0.01 meters wide and white (paint,
spray, white carpet or strong tape). Lines belong to the areas of which they are
boundaries. Distances between lines are measured from their centers. Distances
from a robot are measured from its nearest side to the respective measurement
point, with an assumed radius of 0.09 m. Field Lines The playing area is defined
by four field lines. The two longer field lines are called touch lines. The two
shorter field lines are called goal lines. Halfway Line The field of play is
divided into two halves by a halfway line that runs along the width of the field
and through the center of the field. The halfway line is parallel to the goal
lines. Goal-to-Goal Line The goal-to-goal line runs along the length of the
field, passing through the center of the goals and the field. The goal-to-goal
line is parallel to the touch lines. This line is used to provide adequate
features for the geometry calibration of the vision software and for optional
local localisation of robots. Center Circle The center circle is located at the
center of the field with a diameter of 1 meter. The center is at the crossing of
the halfway line and the goal-to-goal line. Defense Area A defense area is
defined as a rectangle touching the goal lines centrally in front of both goals.
The size of the defense area is 3.6 meters times 1.8 meters for division A and 2
meters times 1 meter for division B, as shown in figures 1 and 2 respectively.
Penalty Mark The penalty mark defines the point from which a team executes a
penalty kick against the opponent goal. It is located on the goal-to-goal line
and 8 meters (division A) or 6 meters (division B) away from the opponent’s goal
center. 2.1.4. Goals Goals must be placed on the center of each goal line and
anchored securely to the field surface. They consist of two 0.16 meters high
vertical side walls joined at the back by a 0.16 meters high vertical rear wall.
The inner face of the goal has to be covered with an energy absorbing material
such as foam to help absorb ball impacts and lessen the speed of deflections.
The inner goal walls are white, the outer goal walls, edges, and tops are black.
The distance between the side walls is 1.8 meters for division A and 1 meter for
division B, and the goal is 0.18 meters deep. The goal walls are 0.02 meters
thick and touch the goal line, but do not overlap or encroach on the field lines
or the field. Figure 3 and figure 4 show these details for division A and
division B respectively. The numbers in the figures below show the distances in
millimeters. goal detail divisionA Figure 3. The goal in detail for division A
goal detail divisionB Figure 4. The goal in detail for division B 2.2. Ball The
ball is a standard orange golf ball. It weights approximately 0.046 kilograms
and its diameter measures 0.043 meters. For official matches, the organizing
committee provides the ball. 2.3. Shared Software The shared software used in
the Small Size League is maintained by the technical committee, though everyone
is encouraged to contribute. The technical committee members however guarantee
that any changes made less than three months before the next RoboCup do not
break compatibility. 2.3.1. Vision Each field is provided with a shared central
vision server and a set of shared cameras. This shared vision equipment uses the
community-maintained SSL-Vision software
(https://github.com/RoboCup-SSL/ssl-vision) to provide localization data to
teams via Ethernet in a packet format that is to be announced by the shared
vision system developers before the competition. Teams need to ensure that their
systems are compatible with the shared vision system output and that their
systems are able to handle the typical properties of real-world sensory data as
provided by the shared vision system (including noise, latency, or occasional
failed detections and misclassifications). The vision patterns on the top of the
robots must adhere to the specifications of SSL-Vision, and must be of the
standard color paper as specified in the SSL-Vision documentation. Besides the
shared vision equipment, teams are not allowed to mount their own cameras or
other external sensors, unless specifically announced or permitted by the
respective competition organizers. SSL-Vision defines an additional tracker
protocol that contains filtered and enriched tracking data. Messages are not
published by SSL-Vision itself, but for example by some automatic referees. It
is meant to be used by the game controller and by teams that do not yet have
their own sophisticated filter. 2.3.2. Game Controller A game is controlled by
the community-maintained ssl-game-controller
(https://github.com/RoboCup-SSL/ssl-game-controller). It is operated by the game
controller operator. The software translates decisions of the referee and the
automatic referee into Ethernet communication signals that are broadcast to the
network. It maintains the state of the game, tracks all events and acts as a
proxy between all participating parties in the game. The game controller has a
network interface for the playing teams. They can automatically change their
keeper id, they can signal a robot substitution intent for the next opportunity,
and they can send an advantage choice for handling game stopping after yellow
cards. 2.3.3. Automatic Referee One or more automatic referee applications can
supervise a game and report offenses to the game controller. At least one
automatic referee is required per game. If more than one automatic referee is
connected to the game controller, a majority vote can be applied. New automatic
referee implementations can be provided, given that the source code is
open-sourced. New implementations must be announced at least three months before
the competition. The technical committee decides if an implementation will be
used or not. The Game Event Table shows which game events an automatic referee
implementation must be able to detect. Individual game events can be disabled
completely or in some automatic referee implementations if both teams and the
technical committee agree. 2.3.4. Remote Control A remote control for each team
can optionally be provided by the tournament organizers. It is a physical device
that allows entering the following commands: Raise a challenge flag Request a
timeout Request robot substitution Request emergency stop Change the keeper id
It may also provide feedback information, like: Number of yellow cards and when
they are due Number of robots currently allowed Number of robots currently on
the field The remote control may only be used by the robot handler. There is
always only one remote control per team, per match. 2.4. Communication Flags The
communication flags are used to avoid gesturing and yelling with the referee
during a match. These flags are responsible for communicating various intents,
such as: timeouts, emergency stops, manual robot substitution and challenges.
The referee or game controller operator has to acknowledge the communication
flag. Any gesturing and yelling will be considered unsporting behavior, punished
by a red card after the first warning. The communication flags are provided by
the organizers of the competition. A remote control software or device can be
provided and replace physical flags. Any other solution that the organizers find
feasible can also be used. 3. Robots 3.1. Number Of Robots A match is played by
two teams, with each team consisting of not more than 11 robots in division A
and not more than 6 robots in division B, one of which may be the keeper. Each
robot must be clearly numbered according to its vision pattern so that the
referee can identify it during the match. The id of the keeper must be chosen
before the match starts (see Choosing Keeper Id). An exception is made during
the Division A group phase, where the number of robots is limited to 8 if at
least one of the two teams prefers it. 3.2. Hardware And Software Constraints
The referee may force a team to remove a robot from the field if it does not
satisfy the rules. Members of the technical committee may also check the
hardware and software constraints of robots at any point of the tournament. If a
team is not able to provide at least one robot that satisfies the rules, the
team is forced to forfeit. 3.2.1. Safety A robot must not pose danger to itself,
another robot, or humans. It must not damage or modify the ball or the field.
The referee has to force a team to remove a robot from the field if he considers
it a potential safety threat. 3.2.2. Shape A robot must fit inside a 0.18 meters
wide and 0.15 meters high cylinder at any point in time. Additionally, the top
of the robot must adhere to the standard pattern size and surface constraints.
3.2.3. Hull Color of Robots To ensure clarity and effective identification
during gameplay, the following rules apply to the coloring of robot hulls:
Bright and dark color scheme: All robots from a team must have interchangeable
hull colors: one bright and one dark. These two colors must be the same for all
robots in a team. The color must vertically cover at least 6 cm of the robot’s
maximum height. Coverage requirements: The robot hull must not be reflective to
avoid interfering with the vision system. The robot hull must avoid colors close
to the ones used at vision pattern and ball. Colors must adhere properly and
remain durable to prevent dropping or peeling during the match. If the color
cover drops, it will be penalized accordingly (see Tipping Over Or Dropping
Parts). 3.2.4. Dribbling Device Dribbling devices that actively exert spin on
the ball, which keep the ball in contact with the robot are permitted under
certain conditions: The dribbling device must not elevate the ball from the
ground Another robot must be able to remove the ball from a robot with the ball.
A robot must not take full control of the ball by removing all of its degrees of
freedom. 80% of the area of the ball when viewed from above has to be outside
the convex hull around the robot. This limitation applies as well to all kicking
devices, even if such infringement is momentary. 3.2.5. Vision Pattern All
participating teams must adhere to the given operating requirements of the
shared vision system. In particular, teams are required to use a certain set of
standardized colors and patterns on top of their robots. To ensure compatibility
with the standardized patterns for the shared vision system, all teams must
ensure that all robots have a flat surface with sufficient space available on
the top side. The color of the robot top must be black or dark grey and have a
matte (non-shiny) finish to reduce glare. The standard vision pattern is
guaranteed to fit within a circle with a radius of 0.085 meters that is linearly
cut off on the front side of the robot to a distance of 0.055 meters from the
centre, as shown in figure 5. Teams must ensure that their robot tops fully
enclose this area. standard pattern2010 Figure 5. Standard Vision Pattern
Dimensions Every robot must have one of the 16 patterns shown in figure 6. No
two robots are allowed to use the same pattern. The center dot color determines
the team and is either blue or yellow (see Choosing Team Colors). The other four
dot colors encode the id of the robot. To ensure that every team uses the same
colors, the organizing committee provides enough colored paper at the
competition. Teams are encouraged to prefer color assignments with ids 0 to 7
because they have been experimentally found more stable, as there is no risk
that the back two dots “color-bleed” into each other. standard colors2010 Figure
6. Standard Vision Pattern Colors 3.2.6. Radio Communication Participants using
wireless communications must notify the organizing committee of the method of
wireless communication, power, and frequency. The organizing committee must be
notified of any change after registration as soon as possible. In order to avoid
interference, a team must be able to select from two carrier frequencies before
the match. The type of wireless communication has to follow legal regulations of
the country where the competition is held. Compliance with local laws is the
responsibility of the competing teams, not the RoboCup Federation. The type of
wireless communication may also be restricted by the local organizing committee.
The local organizing committee will announce any restrictions to the community
as early as possible. Bluetooth is not allowed since it cannot be fixed to
frequency channels. 3.2.7. Autonomy The robotic equipment has to be fully
autonomous. Human operators are not permitted to enter any information to the
system during a match, except in breaks or during a timeout. Additionally,
robots may not be manually moved during any phase of the match, except for
moving robots out of the field, or into the field as described in Robot
Substitution. Disregarding this rule is considered unsporting behavior. 4. Game
Structure 4.1. Impartial Roles To play an official match in the Small Size
League, four impartial roles must be filled: the referee the assistant referee
the game controller operator the vision expert Usually, these roles are filled
by two non-playing teams, with one team providing the referee and the game
controller operator and the other team providing the assistant referee and the
vision expert. The assignment of the roles is up to the organizing committee.
Every participating team is required to be able to provide enough people who are
familiar with these roles. 4.1.1. Referee Each match is controlled by the
referee. He has full authority to enforce the rules of the Small Size League in
connection with the match to which he has been appointed. The referee is
encouraged to use the designated walking area next to the field (see Field
Setup). The referee is assisted by the automatic referee software. The human
referee is allowed to override any decision made by the automatic referee
software. The decisions of the referee regarding facts connected with play are
final. The referee may only change a decision on realizing that it is incorrect
or, at his discretion, on the advice of an assistant referee, provided that he
has not restarted play. The rules do not define all circumstance and their
consequences in every detail. The referee is thus advised to judge in an
adequate way, if the rules are not explicit. The usual procedure is to issue a
warning on the first occurrence and an Unsporting Behavior on repetition. The
referee is not held liable for any kind of injury suffered by an official or
spectator, any damage to property of any kind nor any other loss suffered by an
individual, club, company, association, or other body. The robot handler is the
only team member that may talk to the referee. Duties The referee ensures a safe
match for all humans and robots The referee ensures a fair match according to
the rules of the Small Size League The referee ensures that there is no
interference by unauthorized persons or team members The referee or assistant
referee places the ball for kick-offs and penalties (division A) or after every
stoppage (division B). Subsequently, the referee resumes the match The referee
ensures that the game is started and resumed in time 4.1.2. Assistant Referee
The assistant referee supports the referee wherever he can. He is encouraged to
use the designated walking area next to the field, opposite the referee. No team
members are allowed to talk to the assistant referee. Duties The assistant
referee indicates when misconduct or any other incident has occurred out of the
view of the referee The assistant referee discusses unclear situations with the
referee The referee or assistant referee places the ball for kick-offs and
penalties (division A) or after every stoppage (division B) 4.1.3. Game
Controller Operator During a match, the game controller operator uses the game
controller software as an interface between the referee, the automatic referee
and the team software. No team members are allowed to talk to the game
controller operator, except the robot handler for robot substitution intents.
Duties The game controller operator configures the game controller before the
game begins The game controller operator enters the signals of the referee into
the game controller The game controller operator watches the game event log for
any events that need attention, like detections of an automatic referee or
elapsed timers and notifies the referee 4.1.4. Vision Expert During a match, the
vision expert is in charge of the shared vision system on the field. Team
members are generally advised not to talk to the vision expert, unless they
experience major vision problems. Duties The vision expert checks the vision
hardware and reports any kind of hardware problems to the technical committee
The vision expert monitors the shared vision system during the match and reports
any kind of problems to the referee instantly The vision expert recalibrates the
vision system if the referee deems it necessary 4.2. Team-Specific Roles 4.2.1.
Robot Handler Before the start of the match, every team has to designate one
robot handler. The robot handler represents the team during the match. Duties
The robot handler helps preparing the match. The robot handler asks the referee
for timeouts if necessary. The robot handler can substitute a robot during game
play. The robot handler asks the referee for the permission to substitute a
robot in the next stoppage and, if the referee agrees, substitutes the robot.
The robot handler voices concerns of the team (for example network issues or
vision problems). 4.3. Match Preparation All people that fill a role in the
match (impartial or team-specific) have to be ready at least 10 minutes before
the start of the match to allow the referee to make the following preparations:
4.3.1. Game Result Sheet The referee obtains a game result sheet from the
organizing committee. After the game, the referee fills in the final score,
collects the required signatures and submits the sheet to the organizing
committee. While obtaining the game result sheet, the referee can also take an
official ball and referee equipment such as a whistle or red and yellow cards
(if provided). 4.3.2. Testing The Network The referee ensures that both teams
receive vision data and referee commands. 4.3.3. Choosing Team Colors The
referee asks the robot handlers of the teams about their preferred team color
(either blue or yellow). If the teams agree on a color assignment, the colors
will be used for the entire match. However, if both teams prefer the same color,
the referee assigns the colors by chance. In this case, the teams switch the
colors after the first half of the match as well as after the first half of the
overtime if applicable. 4.3.4. Choosing Robot Hull Colors Teams are encouraged
to choose their preferred robot hull color (dark or bright) before the match. If
both teams choose the same color, the referee assigns the colors by chance
before the game starts. The selected hull color must be used throughout the
game. 4.3.5. Choosing Side And Kick-Off The referee tosses a coin with both
robot handlers. The winning team chooses the goal it will attack in the first
half of the match. The other team takes the kick-off to start the match. 4.3.6.
Choosing Keeper Id The referee asks both robot handlers which robot they will
use as the keeper and forwards this information to the game controller operator.
The keeper id can be changed anytime during the game if the ball is either out
of play or in the opponent’s field half by: Using the game controller network
interface Asking the game controller operator to change it in the game
controller. The game controller operator must not change the keeper id until the
ball is at a valid position. Teams should only ask for a change once the
requirements are met. The game controller operator is responsible for complying
to the rules. If a team does not want to use a keeper, it may select the id of a
robot that is not on the field. 4.4. Game Stages 4.4.1. Overview An official
match of the Small Size League consists of the following stages: Game Stage
Duration First Half 300 seconds of playing time Half-Time Break 300 seconds
pause Second Half 300 seconds of playing time If the match is an elimination
match (draw is not a possible outcome) and the score is even after the regular
game time, the match goes into overtime and the following game stages are added:
Game Stage Duration Pre-Overtime Break 300 seconds of pause Overtime First Half
150 seconds of playing time Overtime Half-Time Break 120 seconds of pause
Overtime Second Half 150 seconds of playing time If the score is even after
overtime has been played, the following stages are added: Game Stage Duration
Pre-Shoot-Out Break 120 seconds of pause Shoot-Out unlimited The match timer is
paused whenever no team is allowed to manipulate the ball. This includes stop,
halt and the preparation states of kick-off and penalty kick. Additionally, it
is paused during ball placement. As a result, the time needed for a match is
much greater than the playing time. 4.4.2. Timeouts The robot handler has to ask
the referee for a timeout. Timeouts are handled like breaks, meaning that both
teams are allowed to make modifications to their software and hardware (see
Autonomy). Any robot that has been physically interacted with during a timeout
must be taken out of the field. It can be brought in again, also during the same
timeout, via the area outlined in Robot Substitution. Each team is allocated 4
timeouts at the beginning of the match. A total of 300 seconds is allowed for
all timeouts. Timeouts may only be taken during a game stoppage. The time is
monitored and recorded by the game controller operator. For example, a team may
take 3 timeouts of 60 seconds duration and thereafter have only one timeout of
up to 120 seconds duration. During overtime, both teams can use 2 timeouts with
a total time of 150 seconds. The number of timeouts and the time not used in
regular game are not added. No timeouts are possible in the shoot-out stage.
4.4.3. Early Termination At A Score Of 10 Before the shoot-out stage, when a
team manages to shoot 10 goals, the match is automatically terminated as soon as
the goal difference is greater than one and the team with more goals is declared
the winner. During the group phase, the number of goals scored is used as
tie-breaker, so the absolute number of goals matter for overall scoring. The
rule applies to all game types for simplicity. 5. Referee Commands An overview
of the referee commands and how they interact with the game state, is given in
Game States. 5.1. Stopping The Game 5.1.1. Stop Definition When the stop command
is issued, all robots have to slow down to less than 1.5 m/s. Additionally, all
robots have to keep at least 0.5 meters distance to the ball and are not allowed
to manipulate the ball. If the ball moves very quickly, it is hard to always
keep the required distance to the ball, especially since the speed of the robots
is limited during stop. Therefore, it is sufficient if it is obvious to the
referee that the robots try their best to follow the distance rule. Usage The
stop command is used to pause the game after the ball crossed the field lines
(including goals) or an offense occurred as well as to prepare the start or
resumption of the game after halt, timeouts and automatic ball placement. The
robot speed limit and the minimum distance to the ball allow the referee or
assistant referee to place the ball safely and without interference. 5.1.2. Halt
Definition When the halt command is issued, no robot is allowed to move or
manipulate the ball. There is a grace period of 2 seconds for the robots to
brake. Usage The halt command allows the referee to interrupt the game
immediately whenever an emergency occurs (for example when a robot gets out of
control). It is also used to recalibrate the vision software during a game if
the vision expert considers it necessary and the referee agrees and for robot
substitution. Additionally, the referee is free to issue the halt command at
will. The halt command is always followed up by stop. Enough preparation time
should be given to teams, before the game is continued. The game controller will
wait for up to 10 seconds after a halt command, but the game can be continued if
robots are prepared already. As a rule of thumb, the game should always be
halted when humans other than the referees are entering the field. 5.2. Ball
Placement Definition After the game was stopped, the ball must be placed on the
appropriate position, depending on the event that occurred. The automatic ball
placement is the preferred way to place the ball at the designated position on
the field by the robots of the teams without human interaction. If this is not
possible, the referee places the ball manually. During manual ball placement,
the game should be in stop to allow robots to prepare for game continuation. No
ball placement is required if all of the following constraints are fulfilled:
The ball is closer than 1m to the designated position. The ball is inside the
field. The ball is at least 0.7m away from any defense area. The ball is
stationary. In this case, the game can be continued as soon as all robots keep
the required distance for stop. A ball is considered placed successfully by the
robots if no more than 30 seconds passed since the placement command there is no
robot within 0.05 meters distance to the ball if the next command is a free kick
for the placing team there is no robot within 0.5 meters distance to the ball if
the next command is a force start the ball is stationary the ball is at a
position within 0.15 meters radius from the requested position No further
commands will be issued by the game controller until the automatic placement is
complete. The game will be continued by the game controller as soon as the ball
is successfully placed, but not earlier than 2 seconds after the ball placement
command has been issued. A failed placement will result in a free kick for the
opposing team. If this team failed to place the ball as well, the ball is placed
by the referee and game continues with the original command. For each team a
ball placement failure counter is incremented on each placement failure and
decremented for successful placements. It can not get negative. The non-placing
team must not interfere the ball placement task. Usage When the ball goes out of
play, the following rules decide, if automatic ball placement is applied: The
referee has to place the ball for all kickoffs and all penalty kicks For a free
kick, the team that brings the ball into play must place the ball For a force
start, a team is drawn by chance and must place the ball The ball must be
visible and must not be inside a field corner, a goal corner or behind the goal,
before the ball placement starts The referee can decide to place the ball
manually at any time The referee can decide to disable automatic ball placement
for the rest of the game. TC/OC must agree with this decision When a teams
placement failure counter reached 5, it is not allowed to place the ball for the
rest of the game half. All free kicks that were a result of the ball leaving the
field, are awarded to the opposing team. For all other rule violations or when
both teams failed to place the ball, the ball is placed by the referee If no
team can place the ball, the ball is placed by the referee or the assistant
referee. Both the referee as well as the assistant referee are advised to use a
so-called ball handler (a long, preferably black stick-like device) to move the
ball. The ball may still be moving when the placement command is issued. The
game commences directly after ball placement. The team receiving the ball may
shoot immediately and leave the opposing team little time to arrange defensive
actions if needed. It is allowed to enter the defense area during ball
placement. Ball placement is mandatory for all teams in division A. Teams in
division B may decide, at any time before or during the game, not to place the
ball for the rest of the game by talking to the referee, who in turn tells the
game controller operator to disable ball placement for this team. In this case,
the team is allowed to bring the ball into play, after the ball was placed by
the opposing team. If the opposing team fails to place the ball or no team can
place the ball, it is placed by the referee or the assistant referee. 5.3.
Resuming The Game After the ball has been placed, the game is resumed using one
of the following commands. 5.3.1. Normal Start Definition For two-staged referee
commands, when normal start is sent, an attacker may manipulate the ball. A
match cannot be resumed directly via normal start. Usage Normal start is used
for kick-offs and penalty kicks. 5.3.2. Kick-Off Definition The ball has to be
placed in the center of the field by the human referee. When the kick-off
command is issued, all robots have to move to their own half of the field
excluding the center circle. However, one robot of the attacking team is also
allowed to be inside the whole center circle. This robot will be referred to as
the kicker. No robot is allowed to touch the ball. When the normal start command
is issued, the kicker is allowed to shoot the ball. A goal may be scored
directly from the kick-off. When the ball is in play, the kicker may not touch
the ball until it has been touched by another robot or the game has been stopped
(see double touch). Also, the restrictions regarding the robot positions are
lifted. Usage Both half times as well as both overtime periods (if needed) start
with a kick-off. Chapter Match Preparation describes how to determine the
attacking team. Additionally, after a goal has been scored, the receiving team
restarts the game with a kick-off. 5.3.3. Free Kick Definition The ball
placement position for a free kick depends on the event that led to the free
kick. This position is valid if there is at least 0.2 meters distance to all
field lines and 1 meter distance to either defense area. If an event requires
the ball to be placed at a position that contravenes this rule, it has to be
placed at the closest valid position instead. When the free kick command is
issued, robots of the attacking team are allowed to approach the ball while
robots of the defending team still have to stay at least 0.5 meters distance
away from the ball (the same distance as in stop). One robot of the attacking
team is allowed to shoot the ball. This robot will be referred to as the kicker.
A goal may be scored directly from the free kick. When the ball is in play, the
kicker may not touch the ball until it has been touched by another robot or the
game has been stopped (see double touch). Also, the restrictions regarding the
robot positions are lifted. Usage Free kicks are used to restart the game after
a foul has occurred. Additionally, goal kicks and corner kicks are mapped to
free kicks. 5.3.4. Force Start Definition When the force start command is
issued, the game is immediately resumed and both teams are allowed to approach
and manipulate the ball again. Usage A neutral forced start is used in
situations where no team is clearly in favor, such as: the game had to be
stopped without a specific reason. both teams are at fault. 5.3.5. Penalty Kick
Definition The procedure of a penalty kick is as follows: The ball is placed by
the human referee on the penalty mark. When the penalty command is issued: The
defending keeper has to move to the goal line between the goal posts and keep
touching the goal line. One attacking robot is allowed to approach the ball but
not allowed to touch the ball. Throughout the penalty kick procedure, all other
robots have to be 1m behind the ball such that they do not interfere the penalty
kick procedure. When the normal start command is issued, the attacker is allowed
to manipulate the ball. The ball has to only move towards the opponent goal, as
measured by its x coordinate in the coordinate system of SSL-Vision. When the
ball is in play, the defending keeper may move freely again. If the ball is
still in play after 10 seconds, the game is stopped. A goal is awarded if: the
ball touches the inner surface of a goal wall or the ground of the goal of the
defending team, starting from when the normal start command is issued. the
defending team commits any foul. The game is continued with a kick-off when a
goal is awarded. A goal is not awarded if: the ball crosses any field lines
outside the goal. the defending keeper touches the ball such that the ball speed
vector changes direction by at least 90 degrees in 2D space. the attacking team
violates any rule. the ball is still in play after 10 seconds. The game is
continued by a goal kick for the defending team when a goal is not awarded. The
restrictions defined for scoring goals, including the ball height limit of 0.15
meters, do not apply here. Other rules like the excessive dribbling limitation
for example do. Additional time is allowed for a penalty kick to be taken at the
end of each half or at the end of periods of overtime. Usage Penalty Kicks are
used to punish unsporting behavior and multiple defenders. 5.4. Ball In And Out
Of Play When the match is stopped, the ball is considered out of play until it
has been brought into play. When the match is resumed, the ball is considered in
play until the next stoppage occurs. The match is resumed when force start has
been issued. the ball moved at least 0.05 meters following a kick-off, free kick
or penalty kick. 10 seconds passed following a kick-off. 5 seconds (Division A)
or 10 seconds (Division B) passed following a free kick. see double touch for
the rationale of the 0.05 meter distance 5.5. Sanctions 5.5.1. Yellow Card
Definition If the yellow card is shown as a result of unsporting behavior, the
referee may decide to immediately halt the match. In this case, the match
continues with a free kick for the other team. Upon receipt of a yellow card,
the number of robots allowed on the field for the penalized team decreases by
one. If, after this decrease, the team has more robots than permitted on the
field, a robot must be taken out. A yellow card does not lead to a stop
automatically. If the ball is in play, the team will have 10 seconds to
automatically remove the robot. If a robot is not taken out within time, the
game is stopped for manual substitution and continues with a Forced Start. The
10 seconds can be extended indefinitely by the other team by sending an advance
choice to the game controller. This rule implies that after receiving a yellow
card, the game might not be automatically stopped. However, the game will be
stopped if the foul that led to the yellow card causes a game stoppage, e.g.
dropping parts. Therefore, if one of those fouls occurred, the team is allowed
to manually remove the robot. No penalty will be given to the team that couldn’t
get the robot out of the field in time. However, in the future there will be a
penalty like this: If the robot gets manually substituted, the ball is placed on
the goal-to-goal line and 1.5 meters away from the teams defense area and the
opposing team gets a free kick. A team cannot score a goal while having more
than the allowed number of robots on the field. After 120 seconds of playing
time (measured by the game controller), the yellow card expires and the number
of allowed robots is increased by one. The team may put a robot back in during
the next opportunity. When a team has two not yet expired yellow cards and
receives another yellow card, this card will be turned into a red card instead.
Usage Yellow cards are used to punish teams that committed multiple fouls.
Yellow cards can also be given by the referee to punish fouls or unsporting
behavior. 5.5.2. Red Card Definition A red card behaves like a yellow card,
except: It does not expire until the end of the game. Usage Red cards are given
by the referee to punish severe fouls or unsporting behavior. For example,
serious violent contact by the robots or disrespectful behavior towards the
referees can result in a red card. 5.5.3. Forced Forfeit Definition A Forced
forfeit means that a team instantly loses the current game with a score of 0 to
10. Usage A team can be forced to forfeit if it is unable to play with at least
one robot that satisfies the rules. A team can only be forced to forfeit in
agreement with members of the technical committee and the organizing committee.
5.5.4. Disqualification Definition A Disqualification means that a team
immediately drops out of the tournament and places last. It will not be eligible
to receive any trophies. Usage A team can be disqualified if members of this
team don’t follow safety guidelines, rules of the venue or commit similarly
severe offenses. A team can only be disqualified in agreement with members of
the technical committee and the organizing committee. 6. Ball Leaves The Field
When the ball leaves the field by fully crossing the field line, the game will
be stopped, the ball will be placed and the game will be restarted depending on
the position of the field line crossing as well as on the team that last touched
the ball. 6.1. Touch Line Crossing Touch lines are the long field lines at both
sides of the playing field. 6.1.1. Throw-In Definition The ball has to be placed
0.2 meters perpendicular to the touch line where the ball crossed the touch
line. Its distance to the goal lines must be at least 0.2 meters. After the ball
has been placed, a free kick is awarded to the opponent of the team that last
touched the ball before it left the field. Usage A throw-in is used to restart
the game after the ball left the field by crossing the touch line. 6.2. Goal
Line Crossing Goal lines are the short field lines at both ends of the playing
field. 6.2.1. Goal Kick Definition The ball has to be placed 0.2 meters from the
closest touch line and 1 meter from the goal line. After the ball has been
placed, a free kick is awarded to the opponent of the team that last touched the
ball before it left the field. Usage A goal kick is used to restart the game
after the ball left the field by crossing the goal line of the team that did not
touch the ball last. In division B, the aimless kick rule might apply instead.
6.2.2. Corner Kick Definition The ball has to be placed 0.2 meters from the
closest touch line and 0.2 meters from the goal line. After the ball has been
placed, a free kick is awarded to the opponent of the team that last touched the
ball before it left the field. Usage A corner kick is used to restart the game
after the ball left the field by crossing the goal line of the team that touched
the ball last. 6.2.3. Aimless Kick (Division B only) Definition The ball has to
be placed at the position from where the ball was kicked (see the free kick
rules for the exact ball position rules). After the ball has been placed, a free
kick is awarded to the opponent of the team that last touched the ball before it
left the field. Usage A kick is aimless when after the ball touched a robot, it
subsequently crossed the halfline and then its opponent’s goal line outside the
goal without touching another robot. A kick-off kick cannot be aimless, as the
ball is located on the halfway line and does therefore not cross it. 7. Scoring
Goals A team scores a goal when the ball fully enters the opponent goal between
the goal posts, provided that: The team did not exceed the allowed number of
robots when the ball entered the goal. The height of the ball did not exceed
0.15 meters after the last touch of the teams robots. The team did not commit
any non stopping foul in the last two seconds before the ball entered the goal.
"The team" refers to the scoring team that is awarded a goal, not the team that
kicked the ball. For example, an own goal is not possible while the opponent
team has too many robots on the field. During penalty kicks, more specific rules
apply. If the goal is considered invalid, the game will be continued as if the
ball crossed the goal line outside the goal. 8. Offenses 8.1. No Progress In
Game If there is no progress in the game for 5 seconds (Division A) or 10
seconds (Division B) while both teams are allowed to manipulate the ball, the
game is stopped and continued by a forced start. 8.2. Double Touch When the ball
is brought into play following a kick-off or free kick, the kicker is not
allowed to touch the ball until it has been touched by another robot or the game
has been stopped. The ball must have moved at least 0.05 meters to be considered
as in play. A double touch results in a stop followed by a free kick from the
same ball position. It is understood that the ball may be bumped by the robot
multiple times over a short distance while the kick is being taken. This is why
a distance of 0.05 meters is used to decide whether a robot violates this rule
or not. Remaining in contact with the ball for more than 0.05 meters also counts
as double touch, even though technically the robot only touched the ball once.
8.3. Unsporting Behavior Unsporting behavior can lead to yellow cards, red
cards, penalty kicks, a forced forfeit or a disqualification. The human referee
chooses an appropriate sanction, depending on the severity of the offense. For
minor infringements, a Yellow Card is adequate, while on more severe
infringements, that gave the team an advantage, a Red Card or Penalty Kick can
be issued. For harder sanctions, the referee is advised to refer to members of
the technical committee or the organizing committee. If the referee is not sure
which sanction to choose, he may confer with the assistant referee and members
of the technical committee or the organizing committee. Some examples of
unsporting behavior are listed below. 8.3.1. Damaging Other Robots It is not
allowed to damage or modify robots of other teams. 8.3.2. Damaging The Field Or
The Ball It is not allowed to damage or modify the field or the ball. 8.3.3.
Disrespect Procedures Not following defined procedures repetitively, like for
example: Robot handler puts a robot on the field, while it is not allowed Robots
do not keep required distance to the ball during stop Robots do not conform to
the positioning rules during a penalty kick and need to be moved or removed
manually 8.3.4. Showing Lack Of Respect A team member must show appropriate
respect to everyone involved in the game. Infringements of this rule include but
are not limited to: insulting the opponent, the referee or other persons holding
an impartial role annoying the referee or other persons holding an impartial
role not obeying the orders of the referee 8.4. Fouls The number of fouls per
team is tracked by a counter. Each foul will increase the counter by one. Every
third increase to the foul counter causes a yellow card to be awarded.
Violations in this section and its subsections increase the foul counter if not
stated otherwise. Regardless, of the prescribed penalties in this section, if a
foul is severe or repeated, the referee can choose to immediately issue a yellow
card or in extreme cases a red card. 8.4.1. Stopping Fouls Fouls in this section
cause the game to stop and then resume with a free kick for the opposite team
from the position where the ball was located when the foul began happening.
Robot Too Close To Opponent Defense Area During stop and free kicks, before the
ball has entered play, all robots have to keep at least 0.2 meters distance to
the opponent defense area. There is a grace period of 2 seconds for the robots
to move away from the opponent defense area. The game is immediately halted
after the second such foul committed by the same team while the game is stopped
or during a free kick, before the ball has entered play. If the first foul is
committed during a free kick, the game is still stopped regularly. The grace
period is restarted after the first foul of the same team. Both fouls count
towards the foul counter. There are no individual fouls per robot. Pushing A
robot pushes an opponent robot if both robots keep contact to the ball or to
each other while the robot exerts force onto the opponent robot, such that both
robots travel towards the opponent robot. If both robots are pushing each other
with similar force, no team is at fault. Ball Holding Robots must not surround
the ball to prevent access by others. Tipping Over Or Dropping Parts A robot
must not tip over, break or drop parts on the field that pose a potential threat
to other robots. A robot violating this rule has to be substituted. Metal parts
(screws for example) as well as larger parts generally pose a potential threat,
very small non-metal parts (for example rubber subwheel rings) don’t. Multiple
Defenders This rule does not use the standard sanctions defined for fouls.
Robots other than the keeper must maintain best-effort to fully stay outside the
own defense area. Infraction of this rule can be rated as unsporting behavior.
If a robot other than the keeper touches the ball while this robot is entirely
inside its own defense area, the game is stopped and a penalty kick is awarded
to the other team. The foul counter is not increased. Boundary Crossing A robot
must not kick the ball over the field boundary such that the ball leaves the
field. Keeper Held Ball The ball must not be kept in the defense area for more
than 5 seconds (Division A) or 10 seconds (Division B). Excessive Dribbling A
robot must not dribble the ball further than 1 meter, measured linearly from the
ball location where the dribbling started. A robot begins dribbling when it
makes contact with the ball and stops dribbling when there is an observable
separation between the ball and the robot. Dribblers can still be used to
dribble large distances with the ball as long as the robot periodically loses
possession, such as kicking the ball ahead of it as human soccer players often
do. 8.4.2. Non Stopping Fouls Fouls in this section do not cause a stop.
Instead, the game continues normally. The same no stop foul cannot be triggered
again until the foul condition has stopped being violated or there has been 2
seconds since the foul was first triggered. This is to allow teams to adjust
their robots' positions, ball speed or any other property that is causing the
violation before being penalized additional times. Attacker Touched Ball In
Opponent Defense Area The ball must not be touched by a robot, while the robot
is partially or fully inside the opponent defense area. Ball Speed A robot must
not accelerate the ball faster than 6.5 meters per second in 3D space. Crashing
At the moment of collision of two robots of different teams, the difference of
the speed vectors of both robots is taken and projected onto the line that is
defined by the position of both robots. If the length of this projection is
greater than 1.5 meters per second, the faster robot committed a foul. If the
absolute robot speed difference is less than 0.3 meters per second, both conduct
a foul. 8.4.3. Fouls While Ball Out Of Play Fouls in this section can only occur
when the ball is out of play. Each foul has a grace period of 2 seconds per team
until it is raised again. If multiple robots commit the same foul within 2
seconds, only the first foul counts. If a robot keeps committing a foul, it will
be punished again after the grace period. Defender Too Close To Ball A robot’s
distance to the ball must be at least 0.5 meters during an opponent kick-off or
free kick. When the foul is committed, the timer of the opponent team for
bringing the ball into play is reset. The human referee may decide to repeat the
kick-off or free kick on significant disturbances. During stop, there is no
automatic sanction for being too close to the ball. The referee may still punish
a team for unsporting behavior by issuing a yellow card if it does not respect
the required distance. See stop for further explanation. Robot Stop Speed A
robot must not move faster than 1.5 meters per second during stop. A violation
of this rule is only counted once per robot and stoppage. There is a grace
period of 2 seconds for the robots to slow down. This rule does not apply to
ball placement. Since the stop command is used for manual ball placement and
robot substitution, the intention of the robot speed limit is to avoid robots
harming the people on the field. Ball Placement Interference During ball
placement, all robots of the non-placing team have to keep at least 0.5 meters
distance to the line between the ball and the placement position (the forbidden
area forms a stadium shape). If a robot of the non-placing team is too close to
the line between the ball and the placement position for more than 2 seconds, it
commits a foul. In this case, 10 seconds are added to the ball placement timer.
Only one interference foul per ball placement phase counts towards the foul
counter, but the placement timer is always incremented. This rule does not cover
all cases of ball placement interference. The referee is encouraged to call
fouls if the non-placing team is obviously interfering with the ball placement.
If a robot keeps interfering the ball placement (for example if it is stuck or
can not move), the human referee is encouraged to stop the placement and place
the ball manually. Excessive Robot Substitutions If a team has used up their
free robot substitution budget, every additional robot substitution is a foul.
The match is resumed with a corner kick for the opponent team. If both teams
committed this foul in the same stop, the match is resumed with the original
command. 9. Robot Substitution Definition Robots are substituted by the robot
handler of the respective team. No other team member is allowed to take robots
out or put robots in. The robot handler should prefer to use long sleeves and
colors that won’t interfere with the vision system. Robots can always be taken
in and out during game play without notifying the referee if all the following
conditions are met: The robot is at least partially inside the field margin. The
robot is at a distance from the halfway line that must not exceed 1 meter. The
ball must be at least 0.5 meters away from the robot. Additionally, robots can
be taken out from any position on request using the procedure below: The robot
handler requests robot substitution at any time. The game controller will halt
the game at the next opportunity. The robot handler may enter the field and
touch robots now. The robot handler takes robots out. The robot handler informs
the referee when done. When both teams finished the robot substitution, the
referee informs the game controller operator. The game controller operator
performs a stop followed by continuing the game. The maximum allowed number of
robots of the team on the field must not be exceeded at any time when putting
robots in. Usage Robots can be substituted for any reason. A substitution grants
the team 20 seconds to take robots out. After that time, a new substitution is
started. Each team has 5 free substitutions per halftime. Every additional
substitution will result in an excessive robot substitutions foul for the team.
A robot substitution intent can be made by: A robot handler by informing the
game controller operator who in turn enters the intent into the game controller.
A robot handler by using the remote control, if provided. A team software by
sending a request to the game controller. The game controller itself if a team
exceeds the maximum number of robots (for example after a team receives a yellow
or red card). If the game was halted due to a substitution intent by a team, at
least one robot must be taken out by this team. A substitution intent can be
revoked unless the game was not already halted for substitution. If a robot
substitution intent for either team is present just before the game would
continue after ball placement, the game controller automatically halts the game.
10. Shoot-Out Definition Both teams alternately attempt to score a goal with a
penalty kick until each team has performed 5 attempts. If both teams have the
same score after those 5 attempts, each team takes another attempt in the same
order as before until the score of the two teams is different. Only up to one
attacking robot and one keeper is allowed per team. During a shoot-out attempt,
the attacking robot and the opponent keeper are the only ones allowed to move
and manipulate the ball. Other robots are not allowed to interfere. If a team is
clearly not able to prepare for a penalty kick, a goal is automatically awarded
to the opposing team. Robots may be substituted between shoot-out attempts. The
new robot may be put in right away. Note that timeouts are not allowed during
shoot-out. If there is no clear progress in determining a winner (after 10
shoot-outs, if both teams time out doing shoot-outs, or if both teams cannot
prepare and execute the penalty kick), the human referee can give both teams a
certain amount of time (like 5min) to change their system. This time can be
applied multiple times, if needed, to eventually determine a winner. Usage
Shoot-Out is used to determine the winner of an elimination match if both teams
scored the same amount of goals in previous game stages. 11. Emergency stop
Definition A team can ask to stop the game immediately after a grace period of
10 seconds or at the next stoppage, whichever happens first regardless of the
current situation. It will receive a yellow card for this and must take a
timeout immediately. If the team is out of timeouts, it is still allowed to
remove robots from the field, but can not use any remaining timeout time. This
rule is supposed to be used in extreme situations only, e.g. a software crash or
when robots are damaging themselves significantly. When the game is stopped due
to this rule, there are three possibilities that may have happened: The grace
period has passed and the game is stopped. The human referee stopped the game
earlier. The game is stopped earlier due to the ball leaving the field or
because of a foul. For these possibilities there are two methods to proceed the
game: For 1 and 2, the game is continued with a free kick for the opposing team.
For 3, the game is continued like after a regular timeout. Usage An emergency
stop intent can be made using communication flags. The referee may stop the game
earlier if there is no promising play in action. 12. Challenge Flags A challenge
flag allows teams to challenge a decision of the referee: If referees decision
was correct, team loses a timeout. If referees decision was incorrect, the
correct decision is applied and the team doesn’t lose a timeout. The flag is
consumed in both cases. Only one ruling may be challenged at a time. The team
must have at least one timeout left before using a challenge flag. Each team
will receive three flags at the start of the game. This rule is inspired by
challenge flags in American football. 13. Rule Changes During Competition Rule
changes between years can have unforeseen consequences. If a rule is found to
cause significant negative impact to the competition, the rules may be adapted
under the following conditions: Only between phases of the competition, like
round-robin and knockout Only for major problems, as a last resort The change
must be approved by all team leaders (by an unanimity vote) Appendix A:
Terminology A.1. Ball Manipulation Shooting and dribbling is considered as
manipulating the ball, the ball accidentally bouncing off the hull is not.
Appendix B: Game States game states Appendix C: Game Events The following game
event table is a compilation of the different game events and their
consequences. It also lists what all Automatic Referee implementations must be
capable of handling. The information shown in this table here may be incomplete.
Please read the sections of the respective events for the full definitions.
Event Applicability Consequence Increments Foul Counter Initiated By While Match
is Running NO_PROGRESS_IN_GAME ball in play Stop → Force Start no game
controller ATTACKER_DOUBLE_TOUCHED_BALL ball in play Stop → Free Kick no auto
referee Ball Leaving the Field POSSIBLE_GOAL ball in play Halt no auto referee
BALL_LEFT_FIELD_TOUCH_LINE ball in play Stop → Free Kick no auto referee
BALL_LEFT_FIELD_GOAL_LINE ball in play Stop → Free Kick no auto referee
AIMLESS_KICK ball in play Stop → Free Kick no auto referee Fouls
DEFENDER_IN_DEFENSE_AREA ball in play Stop → Penalty Kick no auto referee
KEEPER_HELD_BALL ball in play Stop → Free Kick yes game controller
BOUNDARY_CROSSING ball in play Stop → Free Kick yes auto referee
BOT_DRIBBLED_BALL_TOO_FAR ball in play Stop → Free Kick yes auto referee
ATTACKER_TOUCHED_BALL_IN_DEFENSE_AREA ball in play - yes auto referee
BOT_KICKED_BALL_TOO_FAST ball in play - yes auto referee Penalty Kick
PENALTY_KICK_FAILED during Penalty Kick Stop → Free Kick no auto referee, game
controller Always BOT_CRASH_UNIQUE always - yes auto referee BOT_CRASH_DRAWN
always - yes auto referee During Free Kick and While Match is Stop
ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA during Stop and Free Kick Stop → Free Kick
yes auto referee While Match is Stopped Fouls BOT_TOO_FAST_IN_STOP during Stop -
yes auto referee DEFENDER_TOO_CLOSE_TO_KICK_POINT ball out of play timer for
bringing the ball into play is reset yes auto referee Ball Placement
BOT_INTERFERED_PLACEMENT during Ball Placement placement timer increased by 10
seconds yes auto referee PLACEMENT_SUCCEEDED during Ball Placement continue no
auto referee PLACEMENT_FAILED by team in favor during Ball Placement Stop → Free
Kick (div A) / previous command (div B) no game controller PLACEMENT_FAILED by
opponent during Ball Placement Stop no game controller Informational
MULTIPLE_FOULS - Yellow Card no game controller MULTIPLE_CARDS - Red Card no
game controller TOO_MANY_ROBOTS - Stop no game controller INVALID_GOAL - Stop →
Free Kick no game controller BOT_SUBSTITUTION during Stop Halt (after next
stoppage), then Stop no remote control CHALLENGE_FLAG always - no remote control
EMERGENCY_STOP always Halt → Timeout + Yellow Card no remote control Manual GOAL
- Stop → Kick-Off no human referee BOT_PUSHED_BOT always Stop → Free Kick yes
human referee BOT_HELD_BALL_DELIBERATELY ball in play Stop → Free Kick yes human
referee BOT_TIPPED_OVER always Stop → Free Kick yes human referee
UNSPORTING_BEHAVIOR_MINOR always Stop → Yellow Card no human referee
UNSPORTING_BEHAVIOR_MAJOR always Stop → Red Card no human referee A visualized
graph of the game events is stored as graphml and can be viewed at yed-live.
Appendix D: Overview of Timings Situation Div A Time Div B Time Remove robot for
Yellow Card 10 s 10 s penalty kick 10 s 10 s kick-off 10 s 10 s free kick 5 s 10
s Keeper Held Ball inside Defense Area 5 s 10 s No Progress In Game 5 s 10 s
Appendix E: Differences Between Divisions This is a complete list of differences
between division A and division B. Division A plays on a larger field with
larger goals than division B. As a result, a penalty kick is taken from a
greater distance as well. Division A plays with more robots than division B. The
automatic ball placement procedure is mandatory for division A and optional for
division B. The aimless kick rule only applies to division B. Division A has
shorter timeouts in some situations.

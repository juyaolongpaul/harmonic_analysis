\version "2.10.33"

\header {
  title = "43 - Liebster Gott, wann werd' ich sterben"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key e \major
}

soprano = \relative c {
  \repeat volta 2 {
    r4 b''8 b e4 b 
    cis b a4. b16 a 
    gis4 fis r4 b8 a 
    gis4 cis8 b ais fis b4~
    b ais4 b2 
  }
  r4 fis8 gis a4 gis 
  cis4. dis8 bis2 
  r4 cis8 b ais4 b 
  b ais b fis8 fis 
  b4. a8 gis4 cis 
  bis cis2 bis4 
  cis2 r4 e8 b 
  cis4 gis8 a b4. a8 
  gis4 fis8 e dis4 e4~
  e dis4 e2 
}


alto = \relative c {
  \repeat volta 2 {
    r2 r4 gis''8 gis 
    a4 a8 gis gis fis16 e fis8 fis 
    e4 dis r2
    r4 gis8 gis fis4 fis 
    fis fis fis2 
  }
  r4 dis8 e fis4 e8 fis 
  gis fis gis a gis2 
  r4 gis8 gis fis4 fis 
  fis fis fis r4
  r4 fis~ fis e4 
  fis8 gis ais ais gis2 
  gis4 gis8 dis e2~
  e4 e4 dis8 e fis4~
  fis8 e8 dis e b4 cis 
  b2 b 
}


tenor = \relative c {
  \repeat volta 2 {
    r2 r4 e'8 e 
    e4 e cis b 
    b b r2
    r4 e8 dis cis4 dis 
    cis cis dis2 
  }
  r2. e8 dis 
  cis a' gis fis dis2 
  r4 e8 dis cis4 dis 
  cis8 b cis4 dis r
  r dis8 dis b4 cis 
  dis e dis2 
  e4 e8 bis cis4. d8 
  cis2 b2~
  b4 c4 fis, g 
  fis8 e fis4 gis2 
}


baixo = \relative c {
  \repeat volta 2 {
    r2 r4 e8 e 
    a4 e fis dis 
    e b r2
    r4 e4 e8 dis8 cis b 
    fis' e fis4 b,2 
  }
  r2. cis8 dis 
  e dis e fis gis,4 gis'8 fis 
  e2. dis8 e 
  fis4 fis, b r
  r b8 b e dis cis b 
  a gis fisis4 gis2 
  cis r4 cis'8 gis 
  a b cis4 gis dis 
  e a,2 ais4 
  b2 e
}




\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}
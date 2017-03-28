\version "2.10.33"

\header {
  title = "333 - Es woll' uns Gott genädig sein"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key d \major
}

soprano = \relative c'' {
  \partial 4 
  cis4 
  \repeat volta 2 {
    d cis b cis8 d 
    e4 fis e d 
    cis2 r4 e 
    d cis d b 
    a fis8 g a4 g 
  } 
  \alternative {
    { 
      fis2. cis'4 
    }
    { 
      fis,2. a4 
    }
  }
  g fis e fis 
  d e fis cis' 
  d cis8 b a4 b8 cis 
  d4 e cis fis 
  e8 d cis d e4 b 
  a g fis e 
  a b cis8 d e4 
  d cis b d 
  cis b a fis8 g 
  a2 g 
  fis2. 
}


alto = \relative c' {
  \partial 4 
  fis4 
  \repeat volta 2 {
    fis fis8 e d4 a' 
    g fis8 gis a4 b 
    e,2 r4 fis 
    fis e fis d8 e 
    fis2. e4 
  }
  \alternative {
    { 
      d2. e4 
    }
    { 
      d2. fis4 
    }
  }
  e d d8 cis d4 
  d g, d' a' 
  a g fis fis 
  fis8 b b4 a a 
  a a e8 fis g fis 
  e d e4 d cis 
  cis8 a' gis fis e fis g4 
  fis8 b4 ais8 fis4 b~
  b8 a~ a g~ g fis16 e d4~
  d8 e8 fis4 e2 
  d2. 
}


tenor = \relative c' {
  \partial 4 
  ais4 
  \repeat volta 2 {
    b ais b a8 b 
    cis4 d e8 fis gis4 
    cis,2 r4 cis 
    b8 a g4 a b8 cis 
    d2. a4 
  }
  \alternative {
    { 
      a2. ais4 
    }
    { 
      a2. a4 
    }
  }
  a a a a 
  a b8 cis a4 e' 
  d g, d'8 e fis4 
  fis8 d b e e4 d 
  cis8 fis e d cis4 b 
  e, a a a 
  a e'8 d e d cis4~
  cis8 b cis e d4 e 
  e d e8 a, a4 
  d2. cis4 
  a2. 
}


baixo = \relative c {
  \partial 4 
  fis4 
  \repeat volta 2 {
    b, fis' g fis 
    e d cis b 
    a2 r4 ais 
    b e d g 
    fis b a a, 
  }
  \alternative {
    { 
      d2. cis4 
    }
    { 
      d2. d4 
    }
  }
  d8 cis d4 a d8 e 
  fis4 e d cis 
  d e fis8 e d cis 
  b4 e a, d 
  a' a,8 b cis d e d 
  cis b cis a d4 a'8 gis 
  fis4 e8 fis g4 ais, 
  b fis b gis 
  a b cis d8 e 
  fis g a2 a,4 
  d2. 
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
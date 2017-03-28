\version "2.10.33"

\header {
  title = "253 - Ach Gott, vom Himmel sieh' darein"
  composer = "J. S. Bach"
}

global =  {
  \time 4/4 
  \key g \minor
}

soprano = \relative c'' {
  \partial 4 
  a4 
  bes a
  \repeat volta 2 {
    g d' 
    d8 c bes4 a2 
    r4 c bes a 
    g a8 bes16 c bes4 a 
    g2. 
  }
  \alternative {
    {
      a4 bes a
    }
    {
      \partial 4 
      g4
    }
  }
  a f e d8 e 
  f4 g a2 
  r4 a g bes 
  a bes g fis 
  g2. g4 
  d' bes c d 
  c bes a
}


alto = \relative c' {
  \partial 4 
  e4 
  fis8 g4 fis8
  \repeat volta 2 {
    g a bes4 
    a4. g8 fis2 
    r4 fis g f 
    ees8 f g4 g fis 
    d2. 
  }
  \alternative {
    {
      fis4 g fis
    }
    {
      \partial 4 
      d4
    }
  }
  e d cis d 
  c d8 e f2 
  r4 f f8 ees d g~
  g fis8 g4 g8 c, d4 
  d2. d4 
  d g f f8 g 
  a4 g d
}


tenor = \relative c' {
  \partial 4 
  cis4 
  d d
  \repeat volta 2 {
    d d 
    d d d2 
    r4 d d d8 c 
    bes4 ees d4. c8 
    bes2. 
  }
  \alternative {
    {
      d4 d d
    }
    {
      \partial 4 
      bes4
    }
  }
  a a a8 g f g 
  a4 bes c2 
  r4 c c bes8. c16 
  d4 d8 c bes a16 g a4 
  bes2. bes4 
  a d c bes 
  ees e fis
}


baixo = \relative c' {
  \partial 4 
  a4 
  d, d'8 c
  \repeat volta 2 {
    bes4 a8 g 
    fis4 g d2 
    r4 a bes8 c d4 
    ees d8 c d2 
    g,2.
  }
  \alternative {
    {
      d'4 g d'8[ c]
    }
    {
      \partial 4 
      g4
    }
  }
  cis, d a bes 
  a g f2 
  r4 f c' g 
  d' bes ees d 
  g,2. g'4 
  fis g a bes 
  c cis d
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
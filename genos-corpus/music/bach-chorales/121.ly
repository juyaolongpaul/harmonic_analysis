\version "2.10.33"

\header {
  title = "121 - Werde munter, mein Gemüte"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  cis4 
  \repeat volta 2 { 
    d e e 
    d cis b b
    cis d e d8 cis
    b4 b8 a a2
  }
  \alternative {
    {
      \partial 4
      cis4
    }
    {
      b4 cis e8 cis d e
    }
  }
  cis4 cis b2
  d4 e fis e8 fis16 g
  fis4 e8 d d2
  cis4 d e e
  d cis b b
  cis d e a,8 b
  cis4 b8 a a4 a
}

alto = \relative c'' {
  a8[ gis]
  \repeat volta 2 {
    fis4 b8 a gis4
    fis4 e8 fis gis4 gis
    a8 gis a4 a8 gis a4
    a8 fis gis4 e2
  }
  \alternative {
    {
      \partial 4
      a8 gis
    }
    {
      gis4 ais b8 ais b4
    }
  }
  b ais fis2
  b4 b8 a a4 b
  a g8 fis fis2
  e4 a gis b8 a
  a gis a4 e8 fis gis4
  a gis8 fis e gis a4
  a gis e e  
}

tenor = \relative c' {
  fis8[ e]
  \repeat volta 2 {
    d cis b4 cis8 b
    a b cis d e4 e
    e a,8 b cis d e4
    fis e8 d cis2
  }
  \alternative {
    {
      \partial 4
      e4
    }
    {
      e4 e fis e8 d
    }
  }
  g4 fis8 e d2
  fis4 e d8 cis d4
  d8 b cis4 a2
  a4 b8 cis b4 cis
  d8 e fis4 gis e
  e d cis8 d e fis
  e4 d8 cis cis4 cis
}

baixo = \relative c {
  fis4
  \repeat volta 2 {
    b8 a gis fis e4
    fis8 gis a4 e e
    a fis cis b8 a
    d4 e a,2
  }
  \alternative {
    {
      \partial 4
      a'4
    }
    {
      e4 d8 cis b4 g'
    }
  }
  e fis b,2
  b'4 cis d g,
  a a, d2
  a'4 gis8 fis gis4 cis,
  fis e8 dis e4 e
  a,8 gis a b cis b cis d
  e4 e a, a
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
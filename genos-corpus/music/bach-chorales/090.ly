\version "2.10.33"

\header {
  title = "90 - Hast du denn, Jesu, dein Angesicht"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key bes \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    bes4 bes4 f'4 
    d4. c8 bes4 
    a4 bes8 a8 g8 f8 
    bes4 c4 d8 ees8 
    c2 bes4 }    %5
    f'4 ees8 d8 c8 bes8 
  g'2 r4 
  d4 ees4 f4 
  f4 g8 f8 ees8 d8 
  c2 r4   %10
  f,4 g4 a4 
  bes4 c4 d8 ees8 
  c2 bes4 
}

alto = \relative c' {
  \repeat volta 2 {
    f4 f4 f4 
    f4. f8 g4 
    c,4 c4 c4 
    f4 g8 a8 bes8 c8 
    a2 f4 }    %5
    bes4 a4 f4 
  bes2 r4 
  bes4 bes4 a8 g8 
  f4 bes4 g4 
  a2 r4   %10
  f4 c4 c4 
  bes4 f'4 f4 
  f4. ees8 d4
}

tenor = \relative c' {
  \repeat volta 2 {
    d4 d4 c4 
    bes4. c8 d8 e8 
    f4 f4 a,4 
    bes4 ees,4 f4 
    f'4 ees4 d4 }    %5
    d4 ees4 f4 
  ees2 r4
  f4 g4 c,4 
  c4 bes4 bes4 
  f'2 r4   %10
  bes,4 a8 g8 ees'4 
  d4 c4 bes4 
  bes4 a4 f4
}

baixo = \relative c' {
  \repeat volta 2 {
    bes4 bes4 a4 
    bes4. a8 g4 
    f4 c4 ees4 
    d4 c4 bes4 
    f'2 bes,4 }   %5
    bes4 c4 d4 
  ees2 r4
  bes'4 a8 g8 f8 ees8 
  d4 d4 ees4 
  f2 r4   %10
  d4 e4 fis4 
  g4 a4 bes4 
  f2 bes,4
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
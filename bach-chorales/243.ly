\version "2.10.33"

\header {
  title = "243 - Jesu, du mein liebstes Leben"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    g2 a4 
    bes4 c4 d4 
    c2 bes4 
    a2 g4 
    d'4 ees4 f4     %5
    bes,4 c4 d4 
    c2 c4 
    bes2. }
  a2 g4 
  a4 bes4 c4   %10
  bes2 a4 
  g2 f4 
  bes2 a4 
  bes4 c4 d4 
  d2 c4   %15
  d2. 
  a2 d4 
  g,4 a4 bes4 
  a2 a4 
  g2.   %20
  
}

alto = \relative c' {
  \repeat volta 2 {
    d2 a'4~ 
    a8 g8 f8 ees8 d4 
    g4 a4. g8 
    g4 fis4 d4 
    bes'4 a8 g8 f4     %5
    g4 f8 ees8 d4 
    g4 ees8 g8 f8 ees8 
    d2. }
  f2 g4~ 
  g8 f8 e8 d8 c4   %10
  f4 g4. f8~ 
  f4 e4 c4 
  f4 g4 d4~ 
  d8 g8 f8 ees8 d4 
  g8 fis8 g8 bes8 a8 g8   %15
  fis2. 
  fis2 fis4 
  g4 d4 d4 
  ees4 c4 d8 c8 
  b2.   %20
  
}

tenor = \relative c' {
  \repeat volta 2 {
    bes2 d4 
    d4 c4 bes4~ 
    bes4 a4 bes8 d8 
    ees8 a,8 d8 c8 bes4 
    f'4 ees4 d4~     %5
    d4 c4 bes4~ 
    bes4 a8 g8 a4 
    bes2. }
  c2 c4 
  c4 bes4 a4~   %10
  a4 g4 a8 c8 
  d8 g,8 c8 bes8 a4 
  d4 c2 
  bes8 ees8 d8 c8 bes4~ 
  bes8 a8 bes4 ees4   %15
  a,2. 
  d2 d4 
  d4 a4 g4 
  g2 fis4 
  d2.   %20
  
}

baixo = \relative c {
  \repeat volta 2 {
    g4 g'4 fis4 
    g4 a4 bes4 
    e,4 fis4 g4 
    c,4 d4 g,4 
    bes4 c4 d4     %5
    g,4 a4 bes4 
    ees4 c4 f4 
    bes,2. }
  f4 f'4 e4 
  f4 g4 a4   %10
  d,4 e4 f4 
  bes,4 c4 f,4 
  d'4 e4 fis4 
  g4 a4 bes4 
  ees,2.   %15
  d2. 
  d8 d,8 d'8 c8 bes8 a8 
  bes8 a8 g8 fis8 g4 
  c4 a4 d4 
  g,2.   %20
  
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